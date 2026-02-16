# StorageManager + LocalCPUBackend

> **深度**: [MEDIUM] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 1 セッション1）

## 概要

多段ストレージバックエンドを管理するディスパッチャ（StorageManager）と、
L1 CPUメモリキャッシュの実装（LocalCPUBackend）。

**参照**:
- `target/LMCache/lmcache/v1/storage_backend/storage_manager.py`（StorageManager）
- `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py`（LocalCPUBackend）

## StorageManager

### batched_allocate()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:352`

```python
def batched_allocate(
    shapes: Union[torch.Size, list[torch.Size]],
    dtypes: Union[torch.dtype, list[torch.dtype]],
    batch_size: int,              # = num_layers
    fmt: MemoryFormat = KV_2LTD,
    eviction: bool = True,
    busy_loop: bool = True,
) -> Optional[list[MemoryObj]]
```

`allocator_backend`（通常LocalCPUBackend）に委譲。メモリ不足時はNoneを返す。

### batched_put()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:388`

```python
def batched_put(
    keys: Sequence[CacheEngineKey],
    memory_objs: List[MemoryObj],
    transfer_spec=None,
    location: Optional[str] = None,
) -> None
```

**処理フロー**:
1. `allocator_backend`のデータをそのまま利用（コピー不要）
2. `OrderedDict`順に全バックエンド（L1→L2→L3）を走査
3. 異なるallocatorを持つバックエンドには`allocate_and_copy_objects()`で新メモリ確保＋コピー
4. 各バックエンドの`batched_submit_put_task()`を呼び出し
5. 全バックエンド処理後、`ref_count_down()`でrefcount解放

**注意**: `put()`は非推奨（`RuntimeError`を投げる）。`batched_put()`が唯一のエントリポイント。

### contains()

```python
def contains(key: CacheEngineKey) -> Optional[str]
```

全バックエンドを順に検索し、ヒットしたバックエンド名を返す。`batched_contains()`も存在。

## LocalCPUBackend

### submit_put_task()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:141`

**同期実行**（バックグラウンドスレッドなし）。`cpu_lock`下で:
1. 重複チェック: `key in hot_cache` → スキップ
2. `memory_obj.ref_count_up()`
3. `hot_cache[key] = memory_obj`
4. `cache_policy.update_on_put(key)` — Evictionポリシー更新
5. `batched_msg_sender.add_kv_op(ADMIT, key.chunk_hash)` — controller通知（オプション）

### batched_allocate()

MemoryObj（pinned CPUテンソル）のバッチ確保。メモリ不足時はEvictionを試行:
- `eviction=True`: 古いエントリを`cache_policy`に基づき追い出し
- `busy_loop=True`: 確保できるまでEviction→再試行をループ

### hot_cache

`OrderedDict[CacheEngineKey, MemoryObj]`。CachePolicyが管理:
- **FIFO**: `OrderedDict`の先頭から追い出し
- **LRU**: アクセス時にmove_to_end()、先頭から追い出し
- **LFU/MRU**: それぞれの戦略

### メモリ管理

LocalCPUBackendは**メモリアロケータ**としても機能:
- MemoryObjはpinned CPU memory（`page_locked`）
- `MemoryAllocator`がプール管理（事前確保 or 動的確保）
- ref_countで共有管理、0になるとプールに返却

## StorageManager（Retrieve方向）

### batched_get()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:484`

```python
def batched_get(
    keys: List[CacheEngineKey],
    location: Optional[str] = None,
) -> List[Optional[MemoryObj]]
```

指定locationのバックエンドから`batched_get_blocking(keys)`でMemoryObjを取得。
**write-back**: リモートバックエンドから取得した場合、`LocalCPUBackend`が存在すれば自動的にL1にコピー。

### layerwise_batched_get()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:519`

```python
def layerwise_batched_get(
    keys: List[List[CacheEngineKey]],  # [num_layers][num_chunks]
    location: Optional[str] = None,
) -> Generator[Future, None, None]
```

レイヤー単位で非同期取得。各レイヤーの`batched_get_non_blocking()`をasyncio.create_taskで投入し、Futureをyield。
デフォルトlocationは`"LocalCPUBackend"`。

### get_block_mapping()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:908`

チャンクリストを受け取り、各チャンクの所在バックエンドを特定。**prefix match方式**: 各バックエンドの`batched_contains()`で先頭からの連続ヒット数を取得し、残りを次のバックエンドに渡す。

### async_lookup_and_prefetch()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:641`

非同期プリフェッチの中核。LookupServerから呼ばれ、全バックエンドに対してprefix match方式で`batched_async_contains()`→`batched_get_non_blocking()`を実行。結果は`EventManager`にFutureとして登録。

## LocalCPUBackend（Retrieve方向）

### batched_get_non_blocking()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:216`

`cpu_lock`下で`hot_cache[key]`を取得し、`ref_count_up()`してからリストで返す。**同期的に完了**する（CPU上のOrderedDictアクセスのみ）。

### batched_get_blocking()

抽象基底クラス`LMCStorageBackendInterface`のデフォルト実装を継承。個別`get_blocking()`のループ。

## バックエンド登録

StorageManagerの`storage_backends`は`OrderedDict`で登録順=優先度:
```
LocalCPUBackend (L1) → LocalDiskBackend (L2) → RemoteBackend (L3)
```
`batched_put()`は全バックエンドに配布。`batched_get()`は指定locationから取得（リモート→L1 write-back付き）。`get_block_mapping()`はprefix match順。

## 上流・下流

- **上流**: LMCacheEngine（batched_allocate/batched_put/contains等）
- **下流**:
  - LocalCPUBackend（L1 CPUメモリ）
  - LocalDiskBackend（L2 ディスク）
  - RemoteBackend（L3 リモート、Redis/S3等）
- **依存**: CachePolicy（Eviction戦略）、MemoryAllocator（メモリプール）
