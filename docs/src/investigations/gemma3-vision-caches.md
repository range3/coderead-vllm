# Gemma3 ビジョンパイプライン: キャッシュ機構 [MEDIUM] [VERIFIED]

> **最終更新**: 2026-02-11

[gemma3-vision-pipeline.md](gemma3-vision-pipeline.md) で追跡した Gemma3 27B ビジョンパイプライン上には、3つの独立したキャッシュ層が存在する。各キャッシュは異なるステップの重い処理をスキップし、同一画像の再利用や同一プロンプトの再送時に大幅な計算量削減を実現する。

---

## 1. パイプラインとキャッシュの位置関係

```
                      Step 1: API Request
                             │
                      Step 2: chat_template 適用
                             │
                ┌────────────┴────────────────┐
                │  Step 3: Gemma3Processor     │
                │  (CPU, P0 フロントエンド)      │
                │                              │
                │  3a. image_processor          │
                │      resize(896×896)          │
                │      rescale(×1/255)          │  ◀── ProcessorCache ヒット時
                │      normalize(0.5, 0.5)      │      Step 3 全体をスキップ
                │      Pan-and-Scan crop        │
                │  3b. num_crops 取得            │
                │  3c. プロンプト書き換え         │
                │  3d. boi→full_image_seq 展開   │
                │  3e. tokenize                 │
                │  3f. token_type_ids 生成       │
                └────────────┬────────────────┘
                             │
                  pixel_values: (N, 3, 896, 896)
                  prompt_token_ids, mm_hashes
                             │
              ═══════════════╪═══════════════ CPU → GPU (ZMQ IPC)
                             │
                ┌────────────┴────────────────┐
                │  Step 4: SiglipVisionModel   │
                │  (GPU, P1 バックエンド)        │
                │  Conv2d → 4096 patches        │
                │  + position_embedding          │  ◀── EncoderCache ヒット時
                │  SiglipEncoder × 27層          │      Step 4+5+6 をスキップ
                │  post_layernorm               │
                ├───────────────────────────────┤
                │  Step 5: Projector            │
                │  AvgPool2d(k=4) → 256 tokens  │
                │  GemmaRMSNorm                  │
                │  Linear(1152→5376)             │
                ├───────────────────────────────┤
                │  Step 6: split + flatten       │
                └────────────┬────────────────┘
                             │
                  encoder output: (N×256, 5376)
                             │
                ┌────────────┴────────────────┐
                │  Step 7: embed_input_ids     │
                │  text_embeds × normalizer     │
                │  masked_scatter_(mm_embeds)    │  ◀── KVプレフィックスキャッシュ ヒット時
                ├───────────────────────────────┤      プレフィックス一致分の
                │  Step 8: Gemma3 Decoder       │      Step 7+8 をスキップ
                │  62層 Transformer              │      (KV再計算不要)
                └──────────────────────────────┘
```

---

## 2. キャッシュ比較テーブル

| | ProcessorCache | EncoderCache | KVプレフィックスキャッシュ |
|---|---|---|---|
| **場所** | CPU (P0 フロントエンド) | GPU (P1 バックエンド) | GPU (P1 バックエンド) |
| **キャッシュキー** | blake3(model_id, 画像ピクセル, processor_kwargs, tokenizer_kwargs) | `mm_feature.identifier` (= mm_hash or `{lora}:{mm_hash}`) | hash(parent_hash, token_ids, extra_keys) — extra_keysにidentifier含む |
| **保存される値** | HF処理済みテンソル (pixel_values, num_patches) + prompt_updates | エンコーダ出力テンソル (post-Projector, GPU上) | デコーダ各層のKV状態 (KVCacheブロック) |
| **ヒット時にスキップ** | Step 3全体 (CPU前処理) | Step 4+5+6 (エンコーダ+プロジェクタ) | Step 7+8の一部 (プレフィックス分のデコーダ) |
| **Eviction方式** | LRU (サイズベース) | FIFO (RefCount管理) | LRU (ブロック単位) |
| **容量設定** | `mm_processor_cache_gb` (default: 4GB) | `encoder_cache_size` (埋め込み数単位) | KVCacheの一部 (BlockPool管理) |
| **管理クラス** | `MultiModalProcessorOnlyCache` 等4種 | `EncoderCacheManager` + `encoder_cache` dict | `KVCacheManager` (prefix_cache) |

---

## 3. ProcessorCache — CPU側前処理キャッシュ

### ハッシュ計算

**参照**: `target/vllm/vllm/multimodal/hasher.py:50-162`, `target/vllm/vllm/multimodal/processing/processor.py:1299-1363`

```python
MultiModalHasher.hash_kwargs(
    model_id=model_id,          # モデル識別子（例: "google/gemma-3-27b-it"）
    image=PIL_Image,            # 画像データ
    **hf_processor_mm_kwargs,   # HF Processorへの追加引数
    **tokenization_kwargs,      # トークナイザ設定
)
```

ハッシュに投入されるデータ:

| 入力 | シリアライズ方法 |
|------|----------------|
| `model_id` (str) | UTF-8エンコード |
| `image` (PIL.Image) | EXIF ImageID (UUID型) → 16バイト。なければ mode + ピクセルデータ (numpy配列) |
| `image` (MediaWithBytes) | EXIF ImageID → 16バイト。なければ original_bytes |
| `hf_processor_mm_kwargs` (dict) | キーソート → 再帰的シリアライズ |
| `tokenization_kwargs` (dict) | 同上 |

- **ハッシュアルゴリズム**: `VLLM_MM_HASHER_ALGORITHM` 環境変数で設定（`blake3` デフォルト、`sha256`/`sha512` はFIPS準拠用）
- キーはアルファベット順にソートされてから逐次ハッシュに投入される（決定的）

**参照**: `target/vllm/vllm/multimodal/hasher.py:154-162` (hash_kwargs)

### 保存される情報

- **テンソルデータ**: `pixel_values` (形状: `(N, 3, 896, 896)`), `num_patches` (形状: `(num_images,)`)
- **prompt_updates**: プレースホルダー位置情報、展開パターン

**参照**: `target/vllm/vllm/multimodal/cache.py:326-725`

### CPU/GPU

**CPU**。P0フロントエンドプロセスのメモリ上で管理される。

### スキップされる処理

**Step 3 全体**（`Gemma3Processor.__call__()`）:
- 3a: `image_processor` — resize(896×896), rescale(×1/255), normalize(mean=0.5, std=0.5), Pan-and-Scan時のクロップ生成
- 3b: `num_crops` 取得
- 3c: プロンプト書き換え（Pan-and-Scan時のみ）
- 3d: `boi_token` → `full_image_sequence` 展開
- 3e: tokenizer による `token_ids` 変換
- 3f: `token_type_ids` 生成

さらに、Sender/Shm タイプ使用時は **ZMQ IPC でのテンソルデータ転送もスキップ** される（`data=None` で送信）。

**参照**: `target/vllm/vllm/multimodal/processing/processor.py:1513-1596` (_cached_apply_hf_processor)

### キャッシュフロー詳細

```
_cached_apply_hf_processor():
  1. _hash_mm_items()          → MultiModalHashes（各画像のblake3ハッシュ）
  2. _get_cache_missing_items() → 各画像がキャッシュにあるか判定
  3. _apply_hf_processor_main() → キャッシュミスの画像だけHF Processor実行
  4. _merge_mm_kwargs()         → キャッシュ済み + 新規処理の結果をマージ
     ※ マージ前に全ハッシュを touch() して LRU Eviction を防止
```

**参照**: `target/vllm/vllm/multimodal/processing/processor.py:1365-1400` (_get_cache_missing_items)

### 4種の実装

| 実装 | 用途 | 格納先 | ヒット時の動作 |
|------|------|--------|--------------|
| `MultiModalProcessorOnlyCache` | P0完結（IPC無効時） | P0メモリ | テンソル+prompt返却 |
| `MultiModalProcessorSenderCache` | P0→P1（LRUモード） | P0にメタデータのみ | `data=None`で送信、IPC転送省略 |
| `ShmObjectStoreSenderCache` | P0→P1（共有メモリ） | 共有メモリ | 共有メモリ参照を返却 |
| `MultiModalReceiverCache` | P1側（LRUモード） | P1メモリ | P0と同期したLRUでテンソル取得 |

**参照**: `target/vllm/vllm/multimodal/registry.py:284-320` (キャッシュタイプ選択ロジック)

---

## 4. EncoderCache — GPUエンコーダ出力キャッシュ

### キャッシュキー

**参照**: `target/vllm/vllm/v1/engine/input_processor.py:490-506`

```python
identifier = mm_hash                          # 通常
identifier = f"{lora_name}:{mm_hash}"         # LoRA tower connector有効時
```

`mm_hash` は ProcessorCache と同じ blake3 ハッシュ値。LoRA が有効な場合は、同一画像でも LoRA によってエンコーダ出力が変わるため、LoRA名をプレフィックスとして付加する。

### 保存される情報

- **GPU上のテンソル**: SiglipVisionModel + Gemma3MultiModalProjector の出力
  - Gemma3の場合: `(N×256, 5376)` — Projector出力をflattenしたもの
- **論理管理**: `EncoderCacheManager` が RefCount + FIFO で管理
- **物理格納**: `gpu_model_runner.encoder_cache: dict[str, torch.Tensor]`

**参照**: `target/vllm/vllm/v1/core/encoder_cache_manager.py:17-267`, `target/vllm/vllm/v1/worker/gpu_model_runner.py:439`

### CPU/GPU

**GPU**。エンコーダ出力テンソルはGPUメモリ上に保持される。論理管理（RefCount、Eviction判定）はCPU上の `EncoderCacheManager` が行う。

### スキップされる処理

**Step 4 + Step 5 + Step 6**:
- Step 4: **SiglipVisionModel forward** — Conv2d(3→1152) + position_embedding + 27層 Transformer Encoder + post_layernorm
- Step 5: **Gemma3MultiModalProjector forward** — reshape + AvgPool2d(k=4, s=4) + GemmaRMSNorm + Linear(1152→5376)
- Step 6: **split + flatten** — num_patchesに基づく分割と結合

これらはGPU上で最も計算量の大きいビジョン処理であり、特に SiglipEncoder の 27層の双方向 Attention が支配的。

### Scheduler連携

**参照**: `target/vllm/vllm/v1/core/sched/scheduler.py:1060-1215`

```
Scheduler._get_encoder_budget():
  1. 各 mm_feature について:
  2. encoder_cache_manager.check_and_update_cache(req, i) を呼ぶ
     → True: scheduled_encoder_inputs に含めない（スキップ）
     → False: can_allocate() → allocate() → scheduled_encoder_inputs に追加
  3. SchedulerOutput.scheduled_encoder_inputs = {req_id: [input_ids]}
```

GPUModelRunner 側:
```
_execute_mm_encoder():
  → scheduled_encoder_inputs にあるもののみ model.embed_multimodal() 実行
  → 出力を encoder_cache[mm_hash] に格納

_gather_mm_embeddings():
  → 全ての mm_feature について encoder_cache[mm_hash] からスライス取得
  → キャッシュヒットしたものも、ミスして今回計算したものも、同じキャッシュから取得
```

**参照**: `target/vllm/vllm/v1/worker/gpu_model_runner.py:2293-2447` (_execute_mm_encoder), `target/vllm/vllm/v1/worker/gpu_model_runner.py:2449-2527` (_gather_mm_embeddings)

---

## 5. KVプレフィックスキャッシュ — デコーダKV状態キャッシュ

### ブロックハッシュ計算

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:525-552`

```python
BlockHash(
    hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
)
```

`extra_keys` は以下の要素の結合:

```python
extra_keys = lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
```

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:487-522` (generate_block_hash_extra_keys)

#### MM extra keys の生成

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:387-448`

MMトークン（`<image>` token_id=262144）を含むブロックでは、そのブロックに重なる `mm_feature.identifier` が `extra_keys` に追加される。

```
ブロック [start_token_idx, end_token_idx) が
mm_feature の [offset, offset+length) と重なる場合:
  → extra_keys.append(mm_feature.identifier)
```

これにより:
- **同一テキスト・異なる画像** → 異なるブロックハッシュ → キャッシュミス
- **同一テキスト・同一画像** → 同一ブロックハッシュ → キャッシュヒット

### 保存される情報

- **GPUメモリ上のKVCacheブロック**: デコーダ62層分のKey/Value状態
- BlockPool が物理ブロックを管理、prefix_cache がハッシュ→ブロック対応を管理

### CPU/GPU

**GPU**。KV状態はGPUメモリ上のブロックに格納される。ハッシュ計算とブロック対応管理はCPU上の `KVCacheManager` が行う。

### スキップされる処理

**Step 7 + Step 8 の一部**（プレフィックスが一致するトークン分）:
- Step 7: **embed_input_ids** — テキスト埋め込み × normalizer + masked_scatter_(mm_embeds)
- Step 8: **Gemma3 Decoder forward** — 62層 Transformer の KV 計算

プレフィックスキャッシュがヒットすると `num_computed_tokens` が増加し、新規に forward pass が必要なトークン数が減少する。例えば 1000 トークンのプロンプトで 800 トークン分のプレフィックスがヒットすれば、残り 200 トークンだけ計算すればよい。

**参照**: `target/vllm/vllm/v1/core/kv_cache_manager.py:164-204` (get_computed_blocks)

---

## 6. キャッシュの独立性と相互作用

3つのキャッシュは独立に動作する。各キャッシュのヒット/ミスは他のキャッシュの判定に影響しない。

### 典型シナリオ

#### シナリオ1: 初回リクエスト（全ミス）

```
画像A + "この画像は何？"  →  全ステップ実行
  ProcessorCache: MISS → Step 3 実行、結果をキャッシュ
  EncoderCache:   MISS → Step 4+5+6 実行、結果をキャッシュ
  KV Prefix:      MISS → Step 7+8 全トークン実行、KVブロック格納
```

#### シナリオ2: 同一画像・同一プロンプト再送（全ヒット）

```
画像A + "この画像は何？"（2回目）
  ProcessorCache: HIT  → Step 3 スキップ（pixel_values をキャッシュから取得）
  EncoderCache:   HIT  → Step 4+5+6 スキップ（エンコーダ出力をGPUキャッシュから取得）
  KV Prefix:      HIT  → Step 7+8 のプレフィックス分スキップ（KV状態再利用）
```

#### シナリオ3: 同一画像・異なるプロンプト

```
画像A + "この画像を要約して"
  ProcessorCache: HIT  → Step 3 スキップ（同一画像なのでハッシュ一致）
  EncoderCache:   HIT  → Step 4+5+6 スキップ（同一 identifier）
  KV Prefix:      部分HIT → 画像トークン部分（ブロック単位）はヒットする可能性あり
                           テキスト部分は異なるためミス
```

#### シナリオ4: 異なる画像（全ミス）

```
画像B + "この画像は何？"
  ProcessorCache: MISS → ピクセルデータが異なるためハッシュ不一致
  EncoderCache:   MISS → identifier が異なる
  KV Prefix:      MISS → extra_keys の identifier が異なりブロックハッシュ不一致
```

### キャッシュ間のキー共有

3つのキャッシュは同一の **mm_hash**（blake3ハッシュ）を基盤として共有している:

```
MultiModalHasher.hash_kwargs(model_id, image, kwargs...)
        │
        ▼
    mm_hash (blake3 hex digest)
        │
        ├──▶ ProcessorCache のキー（そのまま使用）
        │
        ├──▶ EncoderCache のキー（= identifier = mm_hash or lora:mm_hash）
        │
        └──▶ KV Prefix Cache の extra_keys の一部（= identifier）
```

---

## 7. 主要ファイル参照

| ファイル | 主要クラス/関数 | 行 |
|---------|----------------|-----|
| `target/vllm/vllm/multimodal/hasher.py` | `MultiModalHasher`, `hash_kwargs()`, `serialize_item()` | L50, L154, L52 |
| `target/vllm/vllm/multimodal/cache.py` | `MultiModalProcessorOnlyCache`, `SenderCache`, `ShmCache`, `ReceiverCache` | L326, L379, L437, L614 |
| `target/vllm/vllm/multimodal/processing/processor.py` | `_cached_apply_hf_processor()`, `_hash_mm_items()`, `_get_cache_missing_items()` | L1513, L1299, L1365 |
| `target/vllm/vllm/multimodal/registry.py` | `processor_cache_from_config()` | L305 |
| `target/vllm/vllm/v1/engine/input_processor.py` | `_get_mm_identifier()` | L490 |
| `target/vllm/vllm/v1/core/encoder_cache_manager.py` | `EncoderCacheManager`, `check_and_update_cache()` | L17, L91 |
| `target/vllm/vllm/v1/worker/gpu_model_runner.py` | `encoder_cache`, `_execute_mm_encoder()`, `_gather_mm_embeddings()` | L439, L2293, L2449 |
| `target/vllm/vllm/v1/core/kv_cache_utils.py` | `_gen_mm_extra_hash_keys()`, `generate_block_hash_extra_keys()`, `hash_block_tokens()` | L387, L487, L525 |
| `target/vllm/vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks()` | L164 |
| `target/vllm/vllm/v1/core/sched/scheduler.py` | `_get_encoder_budget()` | L1060 |

## 関連ドキュメント

- [Gemma3 ビジョンパイプライン: 形状フローと数値まとめ](gemma3-vision-pipeline.md)
- [フロントエンド MM処理パス](../components/multimodal/mm-processing.md)
- [バックエンド MM処理パス](../components/multimodal/mm-engine-gpu.md)
- [KVCacheManager](../components/kv-cache-manager/summary.md) — プレフィックスキャッシュの詳細
- [KVCacheManager: プレフィックスキャッシュ](../components/kv-cache-manager/prefix-cache.md)
