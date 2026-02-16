# Phase 2 セッション1: StorageManager + LocalCPUBackend DEEP化

**日付**: 2026-02-16
**Phase**: 2（セッション1）
**目標**: StorageManager + LocalCPUBackend を [MEDIUM]→[DEEP] に昇格

## 調査内容

### MemoryAllocator階層

`target/LMCache/lmcache/v1/memory_management.py`（2,340行）全体を調査。

**発見**:
- **TensorMemoryAllocator**: explicit free list方式。AddressManager（SortedList[FreeBlock]）で仮想アドレス空間管理
  - first-fit確保、4KB境界アライメント、解放時に前後ブロックとcoalesce
  - batched_allocate: 1大ブロック確保→torch.chunk分割（フラグメンテーション軽減）
  - batched_free: アドレス順ソート→隣接ブロック事前coalesce→AddressManager.free()
- **PagedTensorMemoryAllocator**: 固定サイズスロットのdequeでO(1) alloc/free。NIXL/P2P用
- **MixedMemoryAllocator**: LocalCPUBackendのデフォルト。pin_allocator(Tensor/Paged) + buffer_allocator(bytearray)
  - MemoryFormat分岐: KV_*→pin_allocator、BINARY_BUFFER→buffer_allocator
- **MemoryObj**: TensorMemoryObj(ref_count+pin_count管理、group_prefix_sum)とBytesBufferMemoryObj(GC任せ)
- **NUMA対応**: lmc_ops.alloc_pinned_numa_ptr()でNUMA-awareなpinned memory確保
- **MLA first rank**: 最初のrankのみ大容量CPU確保（save_only_first_rank設定）

### CachePolicy 4戦略

`target/LMCache/lmcache/v1/storage_backend/cache_policy/` 4ファイル調査。

**発見**:
- **FIFO**: dict先頭イテレート、update_on_hit/put/evictは全てno-op
- **LRU**: OrderedDict + move_to_end。chunk_hash_to_init_timestampで再利用時間追跡（Prometheus報告）
- **LFU**: SortedDict[freq, dict[key,None]] + key_to_freq。O(log N) update_on_hit。同freq内FIFO
- **MRU**: OrderedDict + reversed()で末尾から追い出し
- **can_evict条件**: `not is_pinned and ref_count == 1`（hot_cacheのみ保持=安全に追い出し可能）

### LocalCPUBackend Evictionループ

- allocate失敗→get_evict_candidates(1)→batched_remove→再確保、のループ
- batched_allocateの特殊処理: Layerwise時にevict_key.split_layers()で全レイヤー一括追い出し
- busy_loop: store=False(デッドロック防止)、retrieve=True(store完了待ち)
- touch_cache(): keys_in_requestを逆順にupdate_on_hit（suffix→prefix順の修正）

### LocalDiskBackend

- StorageBackendInterface直接実装（AllocatorBackendInterfaceではない）
- get_allocator_backend()→local_cpu_backendを返す（メモリ確保はCPU上）
- AsyncPQThreadPoolExecutor: prefetch(0)>delete(1)>put(2)の優先度
- DiskCacheMetadata: path/size/shape/dtype/cached_positions/fmt/pin_count
- O_DIRECT対応: ブロック整列時のみ直接I/O
- 容量Eviction: store時にmax_cache_size超過でcache_policy.get_evict_candidates()→os.remove()

### 独自バックエンド実装インターフェース

- StoragePluginInterface: StorageBackendInterface + コンストラクタに全依存注入
- 必須7+2メソッド、オプション4メソッド（非同期prefetch対応用）
- get_allocator_backend()→local_cpu_backendを返すのが標準パターン

## 成果物

1. `docs/src/lmcache/components/storage-manager/summary.md` — [DEEP]に昇格
2. `docs/src/lmcache/components/storage-manager/memory-allocator.md` — 新規作成
3. `docs/src/lmcache/components/storage-manager/cache-policy.md` — 新規作成
4. `docs/src/lmcache/components/storage-manager/local-disk-backend.md` — 新規作成（独自バックエンド実装ガイド含む）

## 解決した疑問

- LocalCPUBackendの詳細実装（メモリ確保戦略、Eviction実装）
- 独自ストレージバックエンドを作る場合に実装すべきインターフェース
