# BlockPool 詳細

> **深度**: [DEEP]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-11

## 概要

`BlockPool` はKVキャッシュの物理ブロックを管理するクラスである。ブロックの割り当て・解放・プレフィックスキャッシュ索引を一元管理し、LRU Eviction によるメモリ再利用を実現する。3つの内部データ構造（`FreeKVCacheBlockQueue`、`BlockHashToBlockMap`、`KVCacheBlock`）で構成される。

**参照**: `target/vllm/vllm/v1/core/block_pool.py:128`

## KVCacheBlock [DEEP] [VERIFIED]

KVキャッシュブロック1つのメタデータを保持する dataclass。物理メモリ自体は GPU 上にあり、このオブジェクトは CPU 側のメタデータのみを管理する。

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:107`

### フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `block_id` | `int` | 0 〜 `num_gpu_blocks - 1` の一意識別子 |
| `ref_cnt` | `int` | 参照カウント。0なら空きキュー内（Eviction候補） |
| `_block_hash` | `BlockHashWithGroupId \| None` | プレフィックスキャッシュ用ハッシュキー。fullブロックでキャッシュ登録済みの場合のみ設定 |
| `prev_free_block` | `KVCacheBlock \| None` | 空きキューの前ノードポインタ |
| `next_free_block` | `KVCacheBlock \| None` | 空きキューの次ノードポインタ |
| `is_null` | `bool` | null_block フラグ。True の場合は解放・Eviction 対象外 |

### block_hash プロパティ

```python
@block_hash.setter
def block_hash(self, block_hash: BlockHashWithGroupId):
    assert self.block_hash is None  # 二重設定を禁止
    self._block_hash = block_hash

def reset_hash(self):
    self._block_hash = None  # Eviction時にリセット
```

**制約**: setter は `assert self.block_hash is None` で二重設定を防止する。ハッシュのリセットは `reset_hash()` のみで行う。これによりブロックのライフサイクルが「未設定 → 設定 → リセット → 再設定」の順序で制御される。

### ライフサイクル

```
1. 生成: BlockPool.__init__() で全ブロック生成（ref_cnt=0）
2. 割り当て: get_new_blocks() → ref_cnt=1
3. キャッシュ登録: cache_full_blocks() → block_hash 設定
4. 再利用（キャッシュヒット）: touch() → ref_cnt++
5. 解放: free_blocks() → ref_cnt-- → 0なら空きキューへ
6. Eviction: _maybe_evict_cached_block() → hash リセット → 再割り当て
```

## FreeKVCacheBlockQueue [DEEP] [VERIFIED]

空きブロックを LRU 順序で管理する**双方向リンクリスト**。Python 組み込みの `deque` ではなく独自実装を採用している。

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:156`

### なぜ独自実装か

`deque` では中間要素の削除が O(n) であるのに対し、この実装では O(1) で削除できる。`touch()` でキャッシュヒットしたブロックを空きキューの中間から即座に除去する必要があるため、O(1) の `remove()` が不可欠である。

また、Python オブジェクトのアロケーションを行わず、`KVCacheBlock` の `prev_free_block`/`next_free_block` ポインタを直接操作するため、GC 負荷が低い。

### センチネルノード

```
fake_head ⇄ block_0 ⇄ block_1 ⇄ ... ⇄ block_n ⇄ fake_tail
```

- `fake_free_list_head`: `block_id=-1` のダミーノード。先頭の前に配置
- `fake_free_list_tail`: `block_id=-1` のダミーノード。末尾の後に配置
- **目的**: null チェックの分岐を減らし、コードを簡素化

### 操作一覧

| メソッド | 行 | 計算量 | 説明 |
|---------|-----|--------|------|
| `popleft()` | L208 | O(1) | 先頭ブロックを取り出し |
| `popleft_n(n)` | L245 | O(n) | 先頭から n 個を一括取り出し |
| `remove(block)` | L278 | O(1) | 中間のブロックを除去（touch 用） |
| `append(block)` | L298 | O(1) | 末尾にブロックを追加 |
| `append_n(blocks)` | L321 | O(n) | 末尾に複数ブロックを一括追加 |
| `get_all_free_blocks()` | L346 | O(m) | 全空きブロック取得（テスト用） |

### LRU 順序の維持

- **初期状態**: `block_id` 順（0, 1, 2, ...）
- **再挿入時**: `free_blocks()` でブロックが返却される際に**逆順**で追加される
  - 理由: リクエストのブロックチェーンの末尾（最新トークン）は先に evict されるべき。先頭（古いプレフィックス）は他のリクエストと共有される可能性が高いため後回し
  - 逆順操作は `SingleTypeKVCacheManager.free()` 側で実行される（BlockPool 外）

## BlockHashToBlockMap [DEEP] [VERIFIED]

プレフィックスキャッシュのハッシュ→ブロック対応表。

**参照**: `target/vllm/vllm/v1/core/block_pool.py:32`

### データ構造

```python
_cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

**Union 型の最適化**: 大半のハッシュキーには 1 ブロックしか対応しないため、単一ブロックは直接格納し、2つ以上の場合のみ内部 dict に昇格する。これにより内部 dict の GC コストを削減。

### 重複排除なし設計

同一ハッシュのブロックが複数存在しても**重複排除しない**。理由: ブロック ID をリクエストに割り当てた後は追加のみ（append-only）を保証するため。重複排除するとブロック ID が変わり、ブロックテーブルの安定性が崩れる。

### 操作

| メソッド | 行 | 説明 |
|---------|-----|------|
| `get_one_block(key)` | L60 | ハッシュキーに対応する任意の1ブロックを返す。複数あれば先頭 |
| `insert(key, block)` | L73 | ブロックをキャッシュに追加。1→dict 昇格を自動処理 |
| `pop(key, block_id)` | L91 | 特定の block_id を除去。残りがあれば dict を復元 |
| `__len__()` | L121 | ハッシュキー数（ブロック数ではない） |

### insert の分岐

```
key なし       → _cache[key] = block          (単一格納)
key に 1 block → _cache[key] = {id: blk, ...}  (dict 昇格)
key に dict    → dict[block.block_id] = block   (dict 追加)
```

## null_block [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/block_pool.py:174`

### 特性

- **block_id = 0**: 初期化時に空きキューから最初に popleft される
- **is_null = True**: 解放・Eviction の対象外
- **ref_cnt 未管理**: `touch()` や `free_blocks()` で特別にスキップされる

### 用途

| 用途 | 説明 |
|------|------|
| Sliding Window Attention | ウィンドウ外のブロック位置を埋める。物理メモリを消費しない |
| Mamba (align モード) | スキップされたブロック位置のパディング |
| ブロックテーブルの長さ統一 | Attention カーネルにはブロックテーブルの連続性が必要。null_block で長さを揃える |

### ガード条件

```python
# touch() (L383)
if block.ref_cnt == 0 and not block.is_null:
    self.free_block_queue.remove(block)

# free_blocks() (L401-402)
[block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]

# cache_full_blocks() (L260-261)
if blk.is_null:
    continue  # null ブロックはキャッシュしない
```

## ブロック割り当てフロー [DEEP] [VERIFIED]

### get_new_blocks()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:300`

```
get_new_blocks(num_blocks)
  │
  ├─ 空きブロック数チェック → 不足なら ValueError
  │
  ├─ free_block_queue.popleft_n(num_blocks)
  │  └─ LRU 先頭（最古）から取り出し
  │
  ├─ enable_caching の場合:
  │  ├─ _maybe_evict_cached_block(block)  ← ハッシュクリア
  │  ├─ assert block.ref_cnt == 0
  │  ├─ block.ref_cnt += 1
  │  └─ metrics_collector.on_block_allocated(block)  ← サンプリング
  │
  └─ enable_caching でない場合:
     ├─ assert block.ref_cnt == 0
     ├─ block.ref_cnt += 1
     └─ metrics_collector.on_block_allocated(block)
```

**注意**: この関数はキャッシュ検索を行わない。キャッシュヒットの確認は `get_cached_block()` で別途行う。

### get_cached_block()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:182`

```
get_cached_block(block_hash, kv_cache_group_ids)
  │
  ├─ 各 group_id について:
  │  ├─ make_block_hash_with_group_id(block_hash, group_id)
  │  ├─ cached_block_hash_to_block.get_one_block(hash_with_id)
  │  └─ 1つでも miss → None を返す（全グループ一致が必須）
  │
  └─ 全ヒット → list[KVCacheBlock] を返す
```

**All-or-nothing セマンティクス**: 複数の KV キャッシュグループがある場合、全グループでヒットしなければキャッシュミスとして扱う。

## ブロック解放フロー [DEEP] [VERIFIED]

### free_blocks()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:389`

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_list = list(ordered_blocks)           # イテレータを実体化
    for block in blocks_list:
        block.ref_cnt -= 1                       # 参照カウント減少
    self.free_block_queue.append_n(
        [block for block in blocks_list
         if block.ref_cnt == 0 and not block.is_null]  # 0 到達 & 非 null のみ
    )
```

**逆順解放の理由**: 呼び出し元（`SingleTypeKVCacheManager.free()`）がブロックを逆順にして渡す。チェーン末尾（最新トークン）がキューの先頭側に来るため、次回の `get_new_blocks()` で最初に evict される。先頭ブロック（プレフィックス）は末尾側に来るため、プレフィックスキャッシュとして長く生き残る。

### touch()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:372`

プレフィックスキャッシュヒット時に呼ばれ、ブロックの再利用を記録する。

```
touch(blocks)
  │
  ├─ ref_cnt == 0 かつ非 null の場合:
  │  └─ free_block_queue.remove(block)  ← 空きキューから除去
  │
  ├─ ref_cnt += 1
  │
  └─ metrics_collector.on_block_accessed(block)
```

## Eviction メカニズム [DEEP] [VERIFIED]

### _maybe_evict_cached_block()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:332`

新規ブロック割り当て時に、そのブロックがプレフィックスキャッシュに登録されている場合にキャッシュから除去する。

```
_maybe_evict_cached_block(block)
  │
  ├─ metrics_collector.on_block_evicted(block)  ← 先にメトリクス記録
  │
  ├─ block.block_hash is None → return False（キャッシュ未登録）
  │
  ├─ cached_block_hash_to_block.pop(hash, block_id)
  │  └─ None → return False（マップに不在）
  │
  ├─ block.reset_hash()  ← ハッシュをクリア
  │
  ├─ enable_kv_cache_events の場合:
  │  └─ kv_event_queue.append(BlockRemoved(...))
  │
  └─ return True
```

**タイミング**: `get_new_blocks()` 内で `ref_cnt` をインクリメントする**前**に呼ばれる。

### evict_blocks()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:405`

外部（KV コネクタ）から特定の block_id 群を明示的に evict する。`_maybe_evict_cached_block()` を各ブロックに対して呼ぶ。ブロックは空きキューからは除去しない（ハッシュの除去のみ）。

### reset_prefix_cache()

**参照**: `target/vllm/vllm/v1/core/block_pool.py:424`

全プレフィックスキャッシュのリセット。RLHF でモデル重み更新後にキャッシュを無効化する用途。

**前提条件**: 使用中のブロックが null_block のみ（`num_used_blocks == 1`）。条件を満たさない場合は `False` を返して失敗。

```
reset_prefix_cache()
  ├─ 使用中ブロック数 != 1 → return False
  ├─ cached_block_hash_to_block を新規インスタンスで置換
  ├─ 全ブロックの hash をリセット
  ├─ metrics_collector.reset()
  ├─ kv_event_queue.append(AllBlocksCleared())
  └─ return True
```

## KV Cache Events [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/block_pool.py:177-178, 480-490`

KV Transfer（Disaggregated Prefill）連携のためのイベントシステム。

| イベント型 | 発行タイミング | 用途 |
|-----------|--------------|------|
| `BlockStored` | `cache_full_blocks()` | 新規ブロックがキャッシュ登録された |
| `BlockRemoved` | `_maybe_evict_cached_block()` | ブロックがキャッシュから除去された |
| `AllBlocksCleared` | `reset_prefix_cache()` | 全キャッシュがリセットされた |

```python
def take_events(self) -> list[KVCacheEvent]:
    """アトミックにイベントキューを排出"""
    events = self.kv_event_queue
    self.kv_event_queue = []  # 新規リストで置換（参照スワップ）
    return events
```

## キャッシュ使用率 [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/block_pool.py:467`

```python
def get_usage(self) -> float:
    total_gpu_blocks = self.num_gpu_blocks - 1  # null_block を除外
    return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
```

null_block は常に「使用中」だが、使用率の計算からは除外される。

## メトリクス収集 [DEEP] [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/kv_cache_metrics.py:46`

`KVCacheMetricsCollector` はサンプリングベースのブロック滞留メトリクスを収集する。

### BlockMetricsState

**参照**: `target/vllm/vllm/v1/core/kv_cache_metrics.py:16`

個別ブロックのライフサイクル指標:

| フィールド | 説明 |
|-----------|------|
| `birth_time_ns` | 割り当て時刻（`time.monotonic_ns()`） |
| `last_access_ns` | 最終アクセス時刻 |
| `access_history` | アクセス履歴（最大4件、`deque(maxlen=4)`） |

### サンプリング

```python
sample_rate: float = 0.01  # デフォルト1%
def should_sample_block(self) -> bool:
    return random.random() < self.sample_rate
```

全ブロックを追跡するとオーバーヘッドが大きいため、割り当て時に確率的にサンプリングする。サンプリングされたブロックのみ `BlockMetricsState` が生成される。

### イベントフック

| フック | タイミング | 処理 |
|--------|----------|------|
| `on_block_allocated(block)` | `get_new_blocks()` | サンプル判定、`BlockMetricsState` 生成 |
| `on_block_accessed(block)` | `touch()` | `record_access()` 呼び出し |
| `on_block_evicted(block)` | `_maybe_evict_cached_block()` | `KVCacheEvictionEvent` を生成・蓄積 |

Eviction 時に生成される `KVCacheEvictionEvent` には `lifetime_seconds`、`idle_seconds`、`reuse_gaps_seconds` が含まれ、`drain_events()` で一括取得できる。

## BlockPool 初期化 [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/block_pool.py:147`

```python
def __init__(self, num_gpu_blocks, enable_caching, hash_block_size,
             enable_kv_cache_events=False, metrics_collector=None):
    self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
    self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
    self.cached_block_hash_to_block = BlockHashToBlockMap()
    self.null_block = self.free_block_queue.popleft()  # block_id=0
    self.null_block.is_null = True
```

- `hash_block_size`: ハッシュ計算に使うブロックサイズ。通常は実際のブロックサイズと一致するが、Hybrid モデル（異なるブロックサイズの KV キャッシュグループ）では異なる場合がある
- `enable_kv_cache_events`: KV Transfer 連携用のイベント発行を有効化
- `metrics_collector`: サンプリングベースの滞留メトリクス収集（オプション）

## 関連ドキュメント

- [KVCacheManager サマリー](summary.md)
- [プレフィックスキャッシュ詳細](prefix-cache.md)
- [アテンションタイプ別 Manager](attention-type-managers.md)
- [Scheduler](../scheduler/summary.md)
