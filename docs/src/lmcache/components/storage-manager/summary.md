# StorageManager + LocalCPUBackend

> **深度**: [DEEP] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 2 セッション1）

## 概要

多段ストレージバックエンドを管理するディスパッチャ（StorageManager）と、
L1 CPUメモリキャッシュの実装（LocalCPUBackend）。

**参照**:
- `target/LMCache/lmcache/v1/storage_backend/storage_manager.py`（StorageManager）
- `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py`（LocalCPUBackend）
- `target/LMCache/lmcache/v1/storage_backend/abstract_backend.py`（インターフェース定義）

**サブドキュメント**:
- [memory-allocator.md](memory-allocator.md) — メモリアロケータ階層と物理メモリ管理
- [cache-policy.md](cache-policy.md) — Eviction戦略（FIFO/LRU/LFU/MRU）
- [local-disk-backend.md](local-disk-backend.md) — L2ディスクバックエンドと階層化動作

## StorageManager

### バックエンド登録と優先度

`storage_backends`は`OrderedDict`で登録順=優先度:
```
LocalCPUBackend (L1) → LocalDiskBackend (L2) → RemoteBackend (L3)
```

**allocator_backend**: メモリ確保の責務を持つバックエンド。通常は`LocalCPUBackend`（PD有効時は`PDBackend`）。
全バックエンドは`get_allocator_backend()`で自身のallocator元を返す:
- LocalCPUBackend → 自身（`AllocatorBackendInterface`実装）
- LocalDiskBackend → `local_cpu_backend`参照（CPU上に確保してからディスクに書く）
- RemoteBackend → `local_cpu_backend`参照

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:321`

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

`allocator_backend`に委譲。LocalCPUBackendが内部でEviction→再確保のループを行う。

### batched_put()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:388`

**処理フロー**:
1. `allocator_backend`のデータをそのまま利用（コピー不要）
2. `OrderedDict`順に全バックエンド（L1→L2→L3）を走査
3. 異なるallocatorを持つバックエンドには`allocate_and_copy_objects()`で新メモリ確保＋コピー
   - 実際にはLocalDiskBackendもRemoteBackendも`get_allocator_backend()`→LocalCPUBackendなので、同一allocator＝コピー不要
4. 各バックエンドの`batched_submit_put_task()`を呼び出し
5. 全バックエンド処理後、各obj_dictの`ref_count_down()`で解放

**注意**: `put()`は非推奨（`RuntimeError`を投げる）。`batched_put()`が唯一のエントリポイント。

### 運用機能

- **freeze mode**: `_freeze=True`でリモートバックエンドをスキップ（LocalCPUのみ使用）
- **bypass mode**: ヘルスチェック失敗時に特定バックエンドを一時的にバイパス
- **internal_copy_stream**: put時の異なるallocator間コピー用CUDAストリーム

## LocalCPUBackend

`AllocatorBackendInterface`を実装。**メモリ確保**と**キャッシュストレージ**の2つの役割を持つ。

### submit_put_task()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:141`

**同期実行**（バックグラウンドスレッドなし）。`cpu_lock`下で:
1. 重複チェック: `key in hot_cache` → スキップ
2. `memory_obj.ref_count_up()`
3. `hot_cache[key] = memory_obj`
4. `cache_policy.update_on_put(key)` — Evictionポリシー更新
5. `batched_msg_sender.add_kv_op(ADMIT, key.chunk_hash)` — controller通知（オプション）
6. ロック外でon_complete_callback実行

### allocate() / batched_allocate() — Evictionループ

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:426`

```
memory_allocatorに確保試行
  ↓ 失敗
cache_policy.get_evict_candidates(hot_cache, num_candidates=1)
  ↓ 候補あり
batched_remove(evict_keys)  ← hot_cacheから除去 + ref_count_down → allocatorに返却
  ↓
memory_allocatorに再確保試行
  ↓ 失敗 && busy_loop=True
0.1秒待機して再試行（他のstore完了によるメモリ解放を待つ）
```

**batched_allocateの特殊処理**: Layerwise時、1チャンクの全レイヤーをまとめて追い出す
（`evict_key.split_layers(batch_size)`で全レイヤーキーを生成→一括free）。

**busy_loopの用途**:
- store（書き込み）: `busy_loop=False` — 並行storeがデッドロックするため
- retrieve（読み出し）: `busy_loop=True` — storeの完了でメモリが解放されるのを待つ

### hot_cache

`cache_policy.init_mutable_mapping()`が返すマッピング:
- FIFO: `dict`（Python dictは挿入順を保持）
- LRU/MRU: `OrderedDict`
- LFU: `dict`（freq_to_keysで別途管理）

### touch_cache()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:128`

`keys_in_request`を**逆順**にupdate_on_hit()。suffix→prefix順にlookupされたキーを、
prefix→suffix順（正しい時系列順）に修正してアクセス順序を更新。

### contains() with pin

lookup時に`pin=True`で呼ばれると:
1. `hot_cache[key].pin()` → Eviction対象外にマーク
2. `keys_in_request`に追加 → retrieve完了後にtouch_cache()で解除

### initialize_allocator()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:346`

設定に応じてアロケータを選択:
- **P2P有効時**: `PagedCpuGpuMemoryAllocator`（NIXL連携用ページアロケータ）
- **通常時**: `MixedMemoryAllocator`（テンソル用PinMemory + バイナリ用BufferAllocator）
- NUMA対応: GPU→NUMAマッピングでNUMA-awareなpinned memory確保
- MLA first rank: 最初のrankのみ大容量CPU確保
- reserve_cpu_size: システム利用可能メモリから予約サイズを差し引き

## StorageManager（Retrieve方向）

### batched_get()

**参照**: `target/LMCache/lmcache/v1/storage_backend/storage_manager.py:484`

指定locationのバックエンドから`batched_get_blocking(keys)`でMemoryObjを取得。
**write-back**: リモートバックエンドから取得した場合、`LocalCPUBackend`が存在すれば自動的にL1にコピー。

### layerwise_batched_get()

レイヤー単位で非同期取得。各レイヤーの`batched_get_non_blocking()`をasyncio.create_taskで投入し、Futureをyield。

### get_block_mapping()

チャンクリストを受け取り、各チャンクの所在バックエンドを特定。**prefix match方式**: 各バックエンドの`batched_contains()`で先頭からの連続ヒット数を取得し、残りを次のバックエンドに渡す。

### async_lookup_and_prefetch()

非同期プリフェッチの中核。LookupServerから呼ばれ、全バックエンドに対してprefix match方式で`batched_async_contains()`→`batched_get_non_blocking()`を実行。結果は`EventManager`にFutureとして登録。

## バックエンドインターフェース階層

```
StorageBackendInterface (abstract)
├── AllocatorBackendInterface (abstract)  — メモリ確保能力あり
│   └── LocalCPUBackend (concrete)
├── StoragePluginInterface (abstract)     — 独自バックエンド実装用
│   └── (ユーザー定義バックエンド)
├── LocalDiskBackend (concrete)
└── RemoteBackend (concrete)
```

独自バックエンド実装の詳細は [local-disk-backend.md](local-disk-backend.md) 末尾の「独自バックエンド実装ガイド」を参照。

## 上流・下流

- **上流**: LMCacheEngine（batched_allocate/batched_put/contains等）
- **下流**:
  - LocalCPUBackend（L1 CPUメモリ）— [memory-allocator.md](memory-allocator.md), [cache-policy.md](cache-policy.md)
  - LocalDiskBackend（L2 ディスク）— [local-disk-backend.md](local-disk-backend.md)
  - RemoteBackend（L3 リモート、Redis/S3等）
- **依存**: CachePolicy（Eviction戦略）、MemoryAllocator（メモリプール）、EventManager（非同期prefetch）
