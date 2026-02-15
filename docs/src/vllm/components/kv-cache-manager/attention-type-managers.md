# アテンションタイプ別 Manager 詳細

> **深度**: [DEEP]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-11

## 概要

`SingleTypeKVCacheManager` は1種類のアテンションタイプの KV キャッシュ管理ロジックを担当する抽象基底クラスである。アテンションタイプごとにサブクラスが存在し、ブロックの割り当て・解放・プレフィックスキャッシュ検索をそれぞれのセマンティクスに合わせて実装する。

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py`

## spec_manager_map [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:1049`

```python
spec_manager_map = {
    FullAttentionSpec:            FullAttentionManager,
    MLAAttentionSpec:             FullAttentionManager,       # MLA も Full 扱い
    SlidingWindowSpec:            SlidingWindowManager,
    ChunkedLocalAttentionSpec:    ChunkedLocalAttentionManager,
    MambaSpec:                    MambaManager,
    CrossAttentionSpec:           CrossAttentionManager,
    SinkFullAttentionSpec:        SinkFullAttentionManager,
}
```

`get_manager_for_kv_cache_spec(spec, **kwargs)` ファクトリ関数がこのマップから Manager クラスをディスパッチする。

## 基底クラス: SingleTypeKVCacheManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:24`

### 状態管理

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `req_to_blocks` | `defaultdict[str, list[KVCacheBlock]]` | リクエスト ID → 割り当て済みブロックリスト |
| `num_cached_block` | `dict[str, int]` | リクエスト ID → キャッシュ登録済みブロック数。RUNNING リクエストのみ追跡 |
| `block_size` | `int` | 1 ブロックあたりのトークン数。DCP/PCP > 1 の場合は乗算される |

### コンストラクタ

```python
def __init__(self, kv_cache_spec, block_pool, enable_caching,
             kv_cache_group_id, dcp_world_size=1, pcp_world_size=1):
    self.block_size = kv_cache_spec.block_size
    if dcp_world_size * pcp_world_size > 1:
        self.block_size *= dcp_world_size * pcp_world_size
```

DCP（Decode Context Parallelism）/ PCP（Prefill Context Parallelism）ではブロックサイズが並列度倍に拡大される。

### get_num_blocks_to_allocate() [DEEP] [VERIFIED]

**参照**: L73

リクエストに必要な新規ブロック数を算出する。2 つのパスが存在:

```
get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks, ...)
  │
  ├─ Fast-path: request_id in num_cached_block（RUNNING リクエスト）
  │  └─ max(num_required_blocks - num_req_blocks, 0)
  │     ※ Speculative Decoding のリジェクトで num_req_blocks > num_required_blocks もあり得る
  │
  └─ Slow-path: 新規リクエスト（プレフィックスキャッシュヒットあり）
     ├─ num_skipped_tokens = get_num_skipped_tokens(total_computed_tokens)
     ├─ num_skipped_blocks = num_skipped_tokens // block_size
     ├─ num_new_blocks = max(required - max(skipped, local_computed), 0)
     ├─ num_evictable_blocks = Σ(ref_cnt==0 かつ非null)
     │  ← touch() 時にキューから除去されるブロック分を加算
     └─ return num_new_blocks + num_evictable_blocks
```

**Evictable blocks の加算理由**: `new_computed_blocks` 内のブロックが空きキュー内（`ref_cnt == 0`）にある場合、`touch()` で空きキューから除去されるため、実質的に空きブロック数が減る。この分を事前に計上する。

### allocate_new_computed_blocks() [DEEP] [VERIFIED]

**参照**: L137

プレフィックスキャッシュヒットしたブロックをリクエストに追加する。

```
allocate_new_computed_blocks(request_id, new_computed_blocks, ...)
  │
  ├─ RUNNING → assert len(new_computed_blocks) == 0 → return
  │
  └─ 新規リクエスト:
     ├─ num_skipped_blocks 計算
     ├─ スキップ分を new_computed_blocks から除去
     ├─ enable_caching → block_pool.touch(new_computed_blocks)
     ├─ req_blocks に null_block × num_skipped_blocks を追加
     ├─ req_blocks に new_computed_blocks を追加
     ├─ num_cached_block[request_id] = len(req_blocks)
     └─ external_computed_tokens > 0 → 追加ブロック割り当て
```

### allocate_new_blocks() [VERIFIED]

**参照**: L208

```python
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_new_blocks = num_required_blocks - len(req_blocks)
    if num_new_blocks <= 0:
        return []
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)
    return new_blocks  # 新規分のみ返す
```

### cache_blocks() [VERIFIED]

**参照**: L235

```python
def cache_blocks(self, request, num_tokens):
    num_full_blocks = num_tokens // self.block_size
    if num_cached_blocks >= num_full_blocks:
        return  # 既に登録済み
    block_pool.cache_full_blocks(request, req_blocks, num_cached_blocks,
                                  num_full_blocks, block_size, group_id)
    num_cached_block[request_id] = num_full_blocks
```

### free() [VERIFIED]

**参照**: L261

```python
def free(self, request_id):
    req_blocks = self.req_to_blocks.pop(request_id, [])
    ordered_blocks = reversed(req_blocks)  # 逆順でLRU最適化
    self.block_pool.free_blocks(ordered_blocks)
    self.num_cached_block.pop(request_id, None)
```

**逆順の理由**: チェーン末尾（最新トークン）がキューの先頭側に来ることで、次の割り当て時に最初に evict される。先頭ブロック（プレフィックス）は長く残り、共有確率が上がる。

### remove_skipped_blocks() [VERIFIED]

**参照**: L343

```python
def remove_skipped_blocks(self, request_id, total_computed_tokens):
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    if num_skipped_tokens <= 0:
        return  # Full Attention: 何もしない
    # 後方から走査して null_block に遭遇したら停止
    for i in range(num_skipped_blocks - 1, -1, -1):
        if blocks[i] == null_block:
            break  # 前回の呼び出しで既に解放済み
        removed_blocks.append(blocks[i])
        blocks[i] = null_block
    block_pool.free_blocks(removed_blocks)
```

### get_num_skipped_tokens() [VERIFIED]

**参照**: L386

デフォルト実装は `return 0`（全トークンがアテンション対象）。サブクラスでオーバーライド。

## FullAttentionManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:400`

標準的な全トークンアテンション。基底クラスの動作をそのまま継承し、2 つのメソッドのみ実装。

### find_longest_cache_hit()

**参照**: L401

```
左→右にブロックハッシュを走査:
  キャッシュヒット → computed_blocks に追加
  キャッシュミス → break（チェーンが途切れたら以降は必ずミス）

EAGLE 使用時:
  最後のブロックを削除（hidden states 再計算が必要）

alignment_tokens でアライメント調整:
  Hybrid モデルで LCM ブロックサイズの倍数に切り詰め
```

**Downward-closed 性質**: Full Attention では blocks[0..n] がヒットするなら blocks[0..n-1] も必ずヒットする。この性質により、左→右の貪欲スキャンで最適解が得られる。

### get_num_common_prefix_blocks()

**参照**: L450

```python
def get_num_common_prefix_blocks(self, running_request_id):
    for block in blocks:
        if block.ref_cnt == len(self.req_to_blocks):
            num_common_blocks += 1
        else:
            break
    return num_common_blocks
```

**原理**: `ref_cnt == 全リクエスト数` なら、そのブロックは全リクエストで共有されている → 共通プレフィックス。Cascade Attention で共通プレフィックスの再計算をスキップするために使用。

## SlidingWindowManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:461`

Sliding Window Attention 用。ウィンドウ外のトークンの KV キャッシュを解放してメモリを節約する。

### コンストラクタ

```python
def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs):
    super().__init__(kv_cache_spec, **kwargs)
    self.sliding_window = kv_cache_spec.sliding_window
```

### get_num_skipped_tokens()

**参照**: L556

```python
def get_num_skipped_tokens(self, num_computed_tokens):
    return max(0, num_computed_tokens - self.sliding_window + 1)
```

```
例: sliding_window=4, num_computed_tokens=7

Tokens: [0 1 2 3 4 5 6 | 7]
                          ↑ 次に計算するトークン
                  [4 5 6 7]  ← sliding window（サイズ4）
        [0 1 2 3]            ← skipped（4トークン）
```

### find_longest_cache_hit()

**参照**: L466

```
右→左にブロックハッシュを走査:
  キャッシュヒット → computed_blocks[i] にセット、連続カウント++
  キャッシュミス → 連続カウント = 0（リセット）

  連続カウント >= sliding_window_contiguous_blocks:
    末尾をトリミングして break

sliding_window_contiguous_blocks = ceil((window - 1) / block_size)
```

**Right-to-left の理由**: Sliding Window は最新のトークン付近のブロックが重要。右端から連続ヒットを探すことで、ウィンドウ内の有用なキャッシュを効率的に発見する。

**初期値**: `computed_blocks` は `null_block` で埋められ、ヒットした位置のみ実ブロックで置換される。

**制約事項**:
- DCP/PCP 非対応 (`assert dcp_world_size == 1`)
- EAGLE 使用時は `sliding_window_contiguous_blocks += 1`

### get_num_common_prefix_blocks()

**参照**: L584

常に `0` を返す。プレフィックスブロックは全て null_block に置換されているため、Cascade Attention は使用不可。

## ChunkedLocalAttentionManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:594`

チャンク境界でアテンションが分割されるモデル用。各チャンク内のトークンのみが互いにアテンションする。

### コンストラクタ

```python
def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs):
    super().__init__(kv_cache_spec, **kwargs)
    self.attention_chunk_size = kv_cache_spec.attention_chunk_size
```

### get_num_skipped_tokens()

**参照**: L691

```python
def get_num_skipped_tokens(self, num_computed_tokens):
    return (num_computed_tokens // self.attention_chunk_size) * self.attention_chunk_size
```

```
例1: chunk_size=8, computed=13 → skipped=8  (チャンク[0,7]全体)
例2: chunk_size=8, computed=8  → skipped=8  (チャンク[0,7]全体)
例3: chunk_size=8, computed=7  → skipped=0  (まだチャンク内)
```

### find_longest_cache_hit()

**参照**: L599

```
1. local_attention_start_idx = (max_length // chunk_size) * chunk_size
2. computed_blocks = [null_block] × (start_idx // block_size)  ← ウィンドウ外
3. start_idx から max_num_blocks まで左→右スキャン:
   ヒット → append、ミス → break
```

ウィンドウ外のブロックは null_block でパディングし、ウィンドウ内のみ FullAttention と同様の左→右スキャンを行う。

**制約事項**:
- EAGLE 非対応 (`assert use_eagle is False`)
- DCP/PCP 非対応
- 異なるブロックサイズの混在非対応

## MambaManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:744`

Mamba（State Space Model / 線形アテンション）用の Manager。Transformer ベースのアテンションとは根本的に異なり、KV キャッシュではなく「状態」を管理する。

### 2つのキャッシュモード

| モード | 説明 |
|--------|------|
| `none`（デフォルト） | 基底クラスの動作 + speculative blocks 追加 |
| `align` | 最後の状態ブロックのみ追跡、null_block パディング、speculative blocks 再利用 |

### 追加状態（align モード）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `last_state_block_idx` | `dict[str, int]` | 前ステップで割り当てた状態ブロックのインデックス |
| `_allocated_block_reqs` | `set[str]` | ブロック割り当て済みリクエストの集合 |
| `num_speculative_blocks` | `int` | Speculative Decoding 用の余分なブロック数 |

### get_num_skipped_tokens()

**参照**: L967

```python
def get_num_skipped_tokens(self, num_computed_tokens):
    return num_computed_tokens - 1  # 最後の状態のみ必要
```

Mamba は状態の累積的な更新なので、最後のトークンの状態さえあれば以前のトークンの状態は不要。

### find_longest_cache_hit()

**参照**: L756

```
右→左に走査:
  最初のヒットで即座に break
  ヒット位置の前を null_block で埋める
```

最後の状態のみ必要なため、最右のヒット 1 つで十分。

### get_num_blocks_to_allocate()（align モード） [DEEP] [VERIFIED]

**参照**: L832

```
align モード:
  ├─ 既存リクエスト（_allocated_block_reqs に存在）:
  │  └─ 最大 1 ブロック追加（speculative blocks 再利用のため）
  │
  └─ 新規リクエスト:
     └─ 1 + num_speculative_blocks ブロック
```

### allocate_new_blocks()（align モード） [DEEP] [VERIFIED]

**参照**: L885

align モードの割り当ては複雑:

```
1. num_tokens を main model 分に制限（lookahead 除外）
2. last_state_block_idx を記録:
   - 既存: prev_len - 1 - num_speculative_blocks
   - 新規（キャッシュヒット有）: prev_len - 1
3. null_block でスキップ位置をパディング
4. 既存リクエスト: speculative blocks をスキップ位置に移動して再利用
5. 残りの新規ブロックを割り当て
```

### remove_skipped_blocks()（align モード）

**参照**: L804

基底クラスの `remove_skipped_blocks()` に加え、`last_state_block_idx` のブロックも解放する:

```python
if last_state_block_idx < cdiv(num_computed_tokens, block_size) - 1:
    block_pool.free_blocks([blocks[last_state_block_idx]])
    blocks[last_state_block_idx] = null_block
```

2 ステップ前のブロックが不要になるタイミングで解放する。

### free()

**参照**: L961

```python
def free(self, request_id):
    if self.mamba_cache_mode == "align":
        self._allocated_block_reqs.discard(request_id)
        self.last_state_block_idx.pop(request_id, None)
    super().free(request_id)
```

## CrossAttentionManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:976`

エンコーダ-デコーダモデル（Whisper 等）のクロスアテンション用。エンコーダ出力はリクエスト固有（異なる音声/画像入力）のため、プレフィックスキャッシュの恩恵がない。

### 制約

| メソッド | 動作 |
|---------|------|
| `allocate_new_computed_blocks()` | `assert len(new_computed_blocks) == 0` |
| `cache_blocks()` | `raise ValueError` |
| `find_longest_cache_hit()` | `raise NotImplementedError` |
| `get_num_common_prefix_blocks()` | `return 0` |

エンコーダブロックはリクエスト開始時に `num_encoder_tokens` に基づいて静的に割り当てられ、デコード中は変化しない。

## SinkFullAttentionManager [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:1025`

StreamingLLM のための Attention Sink 実装。`FullAttentionManager` を継承し、初期化時に先頭の sink ブロックを事前確保する。

### コンストラクタ

```python
class SinkFullAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: SinkFullAttentionSpec, ...):
        super().__init__(...)
        sink_len = kv_cache_spec.sink_len
        assert sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)
```

**特徴**:
- `sink_len` は `block_size` の倍数でなければならない
- sink ブロックは初期化時に `popleft_n()` で確保され、以降解放されない
- FullAttentionManager の `find_longest_cache_hit()` と `get_num_common_prefix_blocks()` をそのまま使用

## 各 Manager の比較表 [VERIFIED]

| Manager | スキップ計算 | キャッシュ検索 | Cascade | DCP/PCP | EAGLE |
|---------|------------|--------------|---------|---------|-------|
| **FullAttention** | 0（全トークン） | 左→右 | ref_cnt 基準 | 対応 | 対応 |
| **SlidingWindow** | `max(0, n-w+1)` | 右→左（連続） | 非対応 | 非対応 | 対応 |
| **ChunkedLocal** | `(n//c)*c` | null_pad + 左→右 | 非対応 | 非対応 | 非対応 |
| **Mamba** | `n - 1` | 右→左（単一） | 非対応 | 非対応 | - |
| **CrossAttention** | 0 | 非対応 | 非対応 | - | - |
| **SinkFullAttention** | 0 | 左→右 | ref_cnt 基準 | 対応 | 対応 |

## 関連ドキュメント

- [KVCacheManager サマリー](summary.md)
- [BlockPool 詳細](block-pool.md)
- [プレフィックスキャッシュ詳細](prefix-cache.md)
- [Scheduler](../scheduler/summary.md)
