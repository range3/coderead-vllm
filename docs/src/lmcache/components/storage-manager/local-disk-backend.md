# LocalDiskBackend — L2ディスクバックエンド

> **深度**: [DEEP] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 2 セッション1）

## 概要

ローカルディスクにKVキャッシュを永続化するL2バックエンド。
CPU上のMemoryObjをバイト列としてファイルに書き出し/読み出しする。
メモリ確保は`local_cpu_backend`に委譲（自身はAllocatorBackendInterfaceではない）。

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_disk_backend.py`

## アーキテクチャ

```
StorageManager
  ├── LocalCPUBackend (L1)  ← hot_cache + MemoryAllocator
  ├── LocalDiskBackend (L2) ← ファイル永続化 + local_cpu_backendからメモリ借用
  └── RemoteBackend (L3)    ← ネットワーク永続化
```

LocalDiskBackendは`StorageBackendInterface`を直接実装（`AllocatorBackendInterface`ではない）。
`get_allocator_backend()` → `self.local_cpu_backend`を返す。

## Store方向

### submit_put_task()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_disk_backend.py:291`

1. 重複チェック: `exists_in_put_tasks(key)` → スキップ
2. `disk_worker.insert_put_task(key)` — 進行中タスクリストに登録
3. **ディスク容量Eviction**: `disk_lock`下で `current_cache_size + required_size > max_cache_size` の間、
   `cache_policy.get_evict_candidates()` → `batched_remove()` + `os.remove(path)`
4. `memory_obj.ref_count_up()` — 非同期書き込み中の保護
5. `asyncio.run_coroutine_threadsafe()` で `async_save_bytes_to_disk()` を投入

### async_save_bytes_to_disk()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_disk_backend.py:479`

`AsyncPQThreadPoolExecutor`上で実行（max_workers=4のスレッドプール）。

1. `memory_obj.byte_array` → `write_file(buffer, path)`
2. `memory_obj.ref_count_down()` — 参照解放
3. `insert_key(key, size, shape, dtype, fmt)` — `self.dict`にDiskCacheMetadata登録
4. `disk_worker.remove_put_task(key)`
5. `on_complete_callback(key)` 実行（ロック外）

### write_file()

- **通常**: `open(path, "wb").write(buffer)`
- **O_DIRECT**: サイズがディスクブロックサイズの倍数の場合、`os.O_DIRECT`フラグで直接I/O

### DiskCacheMetadata

```python
DiskCacheMetadata(path, size, shape, dtype, cached_positions, fmt, pin_count)
```
MemoryObjとは異なり、ディスク上のファイルパスとメタデータのみ保持。

## Retrieve方向

### get_blocking()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_disk_backend.py:380`

1. `disk_lock`下で`self.dict[key]`からpath/shape/dtype/fmt取得
2. `cache_policy.update_on_hit(key, self.dict)` — アクセス順更新
3. `local_cpu_backend.allocate(shape, dtype, fmt)` — **CPUメモリに確保**
4. `read_file(key, buffer, path)` — ファイルからバッファに読み込み
5. `metadata.cached_positions`をDiskCacheMetadataから復元

### batched_get_non_blocking() — 非同期プリフェッチ

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_disk_backend.py:410`

1. 各キーについて:
   - `local_cpu_backend.allocate()` でCPUメモリ確保
   - `self.dict[key].pin()` — 読み込み中のEviction防止
   - `cache_policy.update_on_hit()` — アクセス順更新
2. `disk_worker.submit_task("prefetch", batched_async_load_bytes_from_disk, ...)`
   - priority=0（最高優先度）でスレッドプールに投入

### batched_async_load_bytes_from_disk()

各ファイルを`read_file()`で読み込み後、`self.dict[key].unpin()`。

### read_file()

- **通常**: `open(path, "rb").readinto(buffer)`
- **O_DIRECT**: ブロック整列時のみ `os.O_DIRECT | os.O_RDONLY`
- FileNotFoundError時: 警告ログ + dictからキー除去

## LocalDiskWorker — 優先度付きスレッドプール

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_disk_backend.py:37`

| タスク種別 | 優先度 | 説明 |
|-----------|--------|------|
| prefetch | 0 (最高) | ディスク→CPUメモリ読み込み |
| delete | 1 | ファイル削除 |
| put | 2 (最低) | CPUメモリ→ディスク書き込み |

`AsyncPQThreadPoolExecutor`（Priority Queue付き非同期スレッドプール）で管理。
prefetchが最優先なのは、retrieve（推論の待ち時間に直結）をstoreより優先するため。

## 容量管理

- `max_cache_size`: `config.max_local_disk_size * 1024^3`（バイト単位）
- `current_cache_size`: 現在のディスク使用量（書き込み時に加算、削除時に減算）
- Eviction: store時にmax超過なら`cache_policy.get_evict_candidates()` → `os.remove()`

**注意**: ディスクフラグメンテーションは未考慮（TODO）。

## 独自ストレージバックエンド実装ガイド

### StoragePluginInterface

**参照**: `target/LMCache/lmcache/v1/storage_backend/abstract_backend.py:394`

独自バックエンドを実装する場合は`StoragePluginInterface`を継承する:

```python
class MyBackend(StoragePluginInterface):
    def __init__(self, dst_device, config, metadata, local_cpu_backend, loop):
        super().__init__(dst_device, config, metadata, local_cpu_backend, loop)
```

### 必須メソッド（StorageBackendInterface由来）

| メソッド | 役割 |
|---------|------|
| `contains(key, pin)` | キーの存在確認。pin=Trueで追い出し保護 |
| `exists_in_put_tasks(key)` | 書き込み進行中のキー確認（重複store防止） |
| `batched_submit_put_task(keys, objs, ...)` | 非同期書き込み投入 |
| `get_blocking(key)` | 同期的なKVデータ取得 |
| `pin(key)` / `unpin(key)` | Eviction保護の制御 |
| `remove(key, force)` | エントリ削除 |
| `get_allocator_backend()` | メモリ確保先（通常`local_cpu_backend`を返す） |
| `close()` | リソース解放 |

### オプショナルメソッド

| メソッド | デフォルト動作 | オーバーライド推奨 |
|---------|---------------|------------------|
| `batched_get_blocking(keys)` | 個別get_blocking()のループ | バッチ取得が効率的な場合 |
| `batched_async_contains(lookup_id, keys, pin)` | NotImplementedError | 非同期prefetch対応時 |
| `batched_get_non_blocking(lookup_id, keys, ...)` | NotImplementedError | 非同期prefetch対応時 |
| `batched_contains(keys, pin)` | 個別contains()のループ（prefix match方式） | バッチ判定が効率的な場合 |
| `touch_cache()` | （定義なし） | LRU等のアクセス順更新が必要な場合 |

### メモリ確保パターン

独自バックエンドからデータを取得する場合:
1. `local_cpu_backend.allocate(shape, dtype, fmt)` でCPUメモリを確保
2. データをMemoryObjの`byte_array`に書き込み（`readinto()`等）
3. `metadata.cached_positions`等のメタデータを復元
4. MemoryObjを返却（呼び出し元がref_count管理）

### on_complete_callback

`batched_submit_put_task()`のオプションパラメータ。
各キーのstore完了時にコールバックを実行（バッチ単位ではなくキー単位）。
他バックエンドへの連鎖store等に使用可能。実装側でcatch/logすること。
