# プレフィックスキャッシュ詳細

> **深度**: [DEEP]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-11

## 概要

プレフィックスキャッシュは、異なるリクエスト間で共通するプロンプトプレフィックスの KV キャッシュブロックを再利用する機構である。トークン列をブロック単位でハッシュ化し、ハッシュチェーン（各ブロックのハッシュが前のブロックのハッシュに依存）を構築することで、プレフィックスの最長一致を効率的に検索する。

## ハッシュチェーン計算 [DEEP] [VERIFIED]

### hash_block_tokens()

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:525`

各ブロックのハッシュは 3 要素のタプルから計算される:

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids,
                      extra_keys=None) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH      # 先頭ブロック用のシード
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

**ハッシュ入力の 3 要素**:

| 要素 | 説明 |
|------|------|
| `parent_block_hash` | 前ブロックのハッシュ（先頭ブロックは `NONE_HASH`） |
| `curr_block_token_ids_tuple` | 現ブロックのトークン ID 列（tuple 化） |
| `extra_keys` | LoRA、マルチモーダル、cache_salt、prompt_embeds（後述） |

### チェーン依存性

```
Block 0: hash(NONE_HASH, tokens[0:B], extra)   → H0
Block 1: hash(H0,        tokens[B:2B], extra)  → H1
Block 2: hash(H1,        tokens[2B:3B], extra) → H2
  ...
```

**なぜチェーンか**: 各ハッシュが全ての先行ブロックに依存するため、プレフィックスが異なれば後続のハッシュも必ず異なる。これにより左から右へのスキャンで「最初のミスで停止」すれば最長プレフィックス一致が得られる。

### NONE_HASH

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:77`

チェーンの起点となるシード値:

```python
def init_none_hash(hash_fn):
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))    # ランダム 32 バイト
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed)) # 決定論的
```

- **PYTHONHASHSEED 未設定**: ランダムシード → プロセス間でハッシュが一致しない
- **PYTHONHASHSEED 設定済み**: 決定論的 → プロセス間でハッシュを共有可能（KV Transfer で必要）
- CBOR ベースのハッシュ関数で PYTHONHASHSEED 未設定の場合は警告が出る

## BlockHash 型 [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:34`

### 型階層

| 型 | 定義 | 用途 |
|----|------|------|
| `BlockHash` | `NewType("BlockHash", bytes)` | ブロック単体のハッシュ値 |
| `BlockHashWithGroupId` | `NewType("BlockHashWithGroupId", bytes)` | ハッシュ + KV キャッシュグループ ID（4 バイト BE） |
| `ExternalBlockHash` | `bytes \| int` | 外部向けハッシュ（後方互換性のための Union） |

### BlockHashWithGroupId のパッキング

```python
def make_block_hash_with_group_id(block_hash, group_id):
    return BlockHashWithGroupId(
        block_hash + group_id.to_bytes(4, "big", signed=False)
    )

def get_block_hash(key):     return BlockHash(key[:-4])
def get_group_id(key):       return int.from_bytes(key[-4:], "big")
```

**設計**: tuple ではなく bytes 結合でパッキングすることで、Python オブジェクト生成を回避し、GC 負荷を低減。

## ハッシュ関数 [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/utils/hashing.py`

4 種類のハッシュ関数が利用可能:

| 名前 | シリアライゼーション | ハッシュ | 出力サイズ | 特徴 |
|------|---------------------|---------|-----------|------|
| `sha256` | pickle | SHA-256 | 32 bytes | Python 依存 |
| `sha256_cbor` | CBOR (canonical) | SHA-256 | 32 bytes | **デフォルト**。言語非依存・再現可能 |
| `xxhash` | pickle | xxh3_128 | 16 bytes | 高速、Python 依存 |
| `xxhash_cbor` | CBOR (canonical) | xxh3_128 | 16 bytes | 高速、言語非依存 |

**デフォルト**: `sha256_cbor` — CBOR の canonical モードにより、PYTHONHASHSEED に依存しないシリアライゼーションが可能。プロセス間でハッシュを共有する KV Transfer に適している。

設定: `vllm_config.cache_config.prefix_caching_hash_algo`

## Extra Keys [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:367`

同一トークン列でも異なる KV キャッシュを持つ場合に、追加情報をハッシュに含める。

### 必要判定

```python
def need_extra_keys(request):
    return (bool(request.mm_features)           # マルチモーダル
            or (request.lora_request is not None) # LoRA
            or (request.cache_salt is not None))  # キャッシュソルト
```

### 各 Extra Key の生成

| 生成関数 | 行 | 内容 | 適用範囲 |
|---------|-----|------|---------|
| `_gen_mm_extra_hash_keys()` | L387 | MM 入力の `identifier` | ブロックと重なる MM 入力のみ |
| `_gen_lora_extra_hash_keys()` | L451 | LoRA アダプタの `lora_name` | 全ブロック共通 |
| `cache_salt` | L508-509 | ユーザー指定のキャッシュソルト | **先頭ブロックのみ** (`start_token_idx == 0`) |
| `_gen_prompt_embeds_extra_hash_keys()` | L466 | プロンプト埋め込みの生テンソルバイト | ブロック範囲分のスライス |

### 結合順序

```python
extra_keys = lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
```

空の場合は `None` を返し、ハッシュ計算の `extra_keys` 引数として渡される。

### マルチモーダル Extra Keys の詳細

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:387`

`_gen_mm_extra_hash_keys()` はブロックの [start, end) トークン範囲と MM 入力の [offset, offset+length) 範囲の重なりを検出する:

```
MM入力: [offset ─────── offset+length]
Block:         [start ──── end]
               ↑ 重なりあり → identifier を extra_keys に追加
```

- `mm_features` は `mm_position.offset` でソート済みと仮定
- `start_mm_idx` で走査位置を追跡し、毎回先頭から検索しない
- `start_mm_idx = -1` は「最後の MM 入力」を示す（生成トークンが増えるデコードフェーズで使用）

## リクエストブロックハッシャー [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:555`

### ファクトリ関数

```python
def get_request_block_hasher(block_size, caching_hash_fn):
    def request_block_hasher(request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        # full ブロックのみハッシュ（不完全ブロックはスキップ）
        if start_token_idx + block_size > request.num_tokens:
            return []
        # ...ハッシュチェーンを走査して新規 full ブロックのハッシュを計算
    return request_block_hasher
```

### 遅延・インクリメンタル計算

1. **初期化時**: `Request.__init__()` で `block_hasher` が渡された場合、即座に `get_hash_new_full_blocks()` を呼び、プロンプトの full ブロック分のハッシュを計算
2. **トークン追加時**: `Request.append_output_token_ids()` で新トークンが追加されるたびに `get_hash_new_full_blocks()` を呼び、新たに full になったブロックのハッシュをインクリメンタルに追加

```python
# Request.__init__()
self.block_hashes = self.get_hash_new_full_blocks()  # 初期ハッシュ

# Request.append_output_token_ids()
self.block_hashes.extend(self.get_hash_new_full_blocks())  # 増分追加
```

**制約**: 不完全ブロック（最後のブロックが block_size 未満）はハッシュされない。これによりプレフィックスキャッシュは常にブロック境界単位で一致する。

### チェーンの継続

```python
prev_block_hash_value = request.block_hashes[-1] if request.block_hashes else None
```

前回計算済みの最後のハッシュを `parent_block_hash` として使い、チェーンを継続する。

## BlockHashListWithBlockSize [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:1571`

Hybrid モデル（異なるブロックサイズの KV キャッシュグループ）でハッシュ粒度を変換するアダプタ。

### 動作原理

```
hash_block_size = 16, target_block_size = 32 の場合:

元のハッシュ:  [H0, H1, H2, H3]  (各16トークン)
変換後:        [H0+H1, H2+H3]     (各32トークン)
                  ↑ bytes 結合
```

```python
def _get_value_at(self, idx):
    base = idx * self.scale_factor   # scale_factor = target / hash
    end = base + self.scale_factor
    merged_hash = self.block_hashes[base]
    for i in range(base + 1, end):
        merged_hash += self.block_hashes[i]  # bytes 結合
    return BlockHash(merged_hash)
```

**遅延評価**: アクセス時にのみ変換を実行。`__getitem__`、`__iter__`、`__len__` をサポートし、通常の `list[BlockHash]` と同じインターフェースで使える。

## Lookup アルゴリズム [DEEP] [VERIFIED]

プレフィックスキャッシュの検索は `SingleTypeKVCacheManager` のサブクラスごとに異なるアルゴリズムを持つ。

### FullAttentionManager: 左→右スキャン

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:401`

```
for block_hash in block_hashes[:max_num_blocks]:
    if cache hit:
        computed_blocks.append(cached_block)
    else:
        break  ← 最初のミスで停止
```

- **Downward-closed 性質**: blocks[0..n] がヒットするなら blocks[0..n-1] も必ずヒットする
- EAGLE 使用時は最後のブロックを削除（hidden states が必要なため）
- `alignment_tokens` でアライメント調整（Hybrid モデルでの LCM ブロックサイズ）

### SlidingWindowManager: 右→左スキャン

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:466`

```
sliding_window_contiguous_blocks = ceil((sliding_window - 1) / block_size)

for i in range(max_num_blocks - 1, -1, -1):  # 右→左
    if cache hit:
        computed_blocks[i] = cached_block
        num_contiguous_blocks += 1
        if num_contiguous_blocks >= required:
            break  ← ウィンドウ分の連続ブロック確保
    else:
        num_contiguous_blocks = 0  ← 連続性リセット
```

- **連続ブロックが必須**: Sliding Window Attention はウィンドウ内の連続したトークンにのみアテンションするため、不連続なブロックは使えない
- `computed_blocks` は初期値として `null_block` で埋められ、ヒットした位置のみ実ブロックで置換される
- アライメントチェック: 右端のブロックがアライメント境界に合わない場合はスキップ

### ChunkedLocalAttentionManager

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:594`

チャンク境界でアテンションが分割されるモデル用。ウィンドウ外のブロックは null_block でパディングし、ウィンドウ内のみ左→右スキャンで検索する。

### MambaManager: 右→左、単一一致

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:744`

Mamba（線形アテンション）は最後の状態のみが必要なため、最初のヒットで即座に停止する。

### HybridKVCacheCoordinator: 反復固定点

**参照**: `target/vllm/vllm/v1/core/kv_cache_coordinator.py:448`

複数のアテンションタイプが混在する Hybrid モデルでは、各グループの最長ヒット長が相互に制約し合う:

```
hit_length = max_cache_hit_length

while True:
    curr_hit_length = hit_length
    for each attention_group:
        if is_full_attention and cached:
            # Downward-closed: 既存結果を再利用
            curr_hit_length = (curr_hit_length // block_size) * block_size
        else:
            hit = find_longest_cache_hit(curr_hit_length)
            curr_hit_length = len(hit) * block_size

    if curr_hit_length >= hit_length:
        break  ← 収束（もう減らない）
    hit_length = curr_hit_length

    if is_simple_hybrid:  # FullAttn + 1種のみ
        break  ← 1回で十分
```

**収束保証**: `hit_length` は単調減少するため、有限回で収束する。
**最適化**: Full Attention は downward-closed なので、他グループの結果に合わせてカットするだけでよい。2グループの simple hybrid ケースでは 1 イテレーションで確定。

## キャッシュ登録 [DEEP] [VERIFIED]

### cache_full_blocks()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:209`

計算済みの full ブロックをプレフィックスキャッシュに登録する。

```
cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, ...)
  │
  ├─ num_cached_blocks >= num_full_blocks → return（既に登録済み）
  │
  ├─ new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
  │
  ├─ block_size == hash_block_size の場合:
  │  └─ block_hashes = request.block_hashes（直接使用）
  │
  ├─ block_size != hash_block_size の場合:
  │  └─ block_hashes = BlockHashListWithBlockSize(...)（粒度変換）
  │
  └─ 各 new_full_block について:
     ├─ is_null → skip
     ├─ assert block_hash is None（二重登録防止）
     ├─ block_hash_with_group_id を生成
     ├─ blk.block_hash = block_hash_with_group_id
     ├─ cached_block_hash_to_block.insert(...)
     └─ enable_kv_cache_events → BlockStored イベント発行
```

### num_cached_block トラッキング

`SingleTypeKVCacheManager` が `num_cached_block[request_id]` で各リクエストの登録済みブロック数を追跡する。`cache_blocks()` 呼び出し時に `num_cached_blocks >= num_full_blocks` なら何もせず、新たに full になったブロックのみを登録する。

## データフロー全体 [VERIFIED]

```
EngineCore.__init__()
  └─ init_none_hash(hash_fn)        ← グローバル NONE_HASH 初期化
  └─ get_request_block_hasher(...)   ← ハッシャークロージャ生成

Request.__init__(block_hasher=...)
  └─ block_hashes = get_hash_new_full_blocks()  ← プロンプトの full ブロックを即時ハッシュ

Request.append_output_token_ids()
  └─ block_hashes.extend(get_hash_new_full_blocks())  ← 増分ハッシュ

Scheduler.schedule()
  ├─ kv_cache_manager.get_computed_blocks(request)
  │  └─ coordinator.find_longest_cache_hit(request.block_hashes, max_length)
  │     └─ block_pool.get_cached_block(hash, group_ids)
  │        └─ BlockHashToBlockMap.get_one_block(hash_with_group_id)
  │
  └─ kv_cache_manager.allocate_slots(request, ...)
     └─ coordinator.cache_blocks(request, num_tokens_to_cache)
        └─ block_pool.cache_full_blocks(request, blocks, ...)
           └─ BlockHashToBlockMap.insert(hash_with_group_id, block)
```

## 関連ドキュメント

- [KVCacheManager サマリー](summary.md)
- [BlockPool 詳細](block-pool.md)
- [アテンションタイプ別 Manager](attention-type-managers.md)
- [用語集](../../glossary.md)
