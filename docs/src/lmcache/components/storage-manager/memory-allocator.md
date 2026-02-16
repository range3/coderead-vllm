# メモリアロケータ階層

> **深度**: [DEEP] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 2 セッション1）

## 概要

LMCacheのメモリ管理は、事前確保された大きなバッファ上で仮想アドレス空間管理を行う
カスタムアロケータで実現されている。pinned CPUメモリ上に確保することで、
GPU⇔CPU間のDMA転送を高速化する。

**参照**: `target/LMCache/lmcache/v1/memory_management.py`

## アロケータ階層

```
MemoryAllocatorInterface (abstract)
├── TensorMemoryAllocator         ← explicit free list方式
├── PagedTensorMemoryAllocator    ← ページ単位の固定サイズスロット
├── BufferAllocator               ← バイト配列用（GC任せ）
├── HostMemoryAllocator           ← 非pinned CPU + 内部委譲
├── PinMemoryAllocator            ← pinned CPU + 内部委譲
├── MixedMemoryAllocator          ← テンソル用Pin + バイナリ用Buffer
├── GPUMemoryAllocator            ← GPU VRAM + 内部委譲
├── CuFileMemoryAllocator         ← GPUDirect Storage対応
├── PagedCpuGpuMemoryAllocator   ← CPU+GPU両方のページアロケータ（NIXL/P2P用）
└── AdHocMemoryAllocator          ← テスト用ダミー
```

## TensorMemoryAllocator — explicit free list方式

**参照**: `target/LMCache/lmcache/v1/memory_management.py:1135`

事前確保されたフラットテンソル(`buffer`)上で、AddressManagerが仮想アドレス空間を管理。

### AddressManager

**参照**: `target/LMCache/lmcache/v1/memory_management.py:903`

- **データ構造**: `SortedList[FreeBlock]`（`sortedcontainers`ライブラリ）
- **FreeBlock**: `(start, size)` タプル。startでソート
- **アライメント**: デフォルト4,096バイト（`ALIGN_BYTES`）
- **確保**: first-fit方式。ソートリストを先頭から走査し最初に十分な空きブロックを選択
- **解放**: bisect_leftで挿入位置を特定し、前後のブロックとcoalesce（結合）
- **sbrk()**: アドレス空間の動的拡張（LazyAllocator向け）
- **スレッドセーフ**: `@synchronized("_lock")`デコレータで全操作をロック保護

### allocate() の動作

```python
aligned_size = (raw_size + 4095) & ~4095  # 4KB境界に切り上げ
# SortedListからfirst-fitで空きブロックを探索
# ブロックを分割: [確保分 | 残余→フリーリストに戻す]
# buffer[start:start+raw_size] をraw_dataとしてTensorMemoryObjを生成
```

### batched_allocate() の最適化

通常のallocateはブロックごとに個別確保だが、batched_allocateは
`unit_aligned_size * batch_size` の**1つの大きなブロック**として確保し、
`torch.chunk()`で分割。これによりフラグメンテーションを軽減。

### free() / batched_free()

- `free()`: AddressManager.free()でブロックを返却→前後のFreeBlockとcoalesce
- `batched_free()`: メモリブロックをアドレス順にソートし、隣接ブロックを事前にcoalesceしてから
  AddressManager.free()を呼ぶ。フリーリスト操作回数を削減

## PagedTensorMemoryAllocator — ページスロット方式

**参照**: `target/LMCache/lmcache/v1/memory_management.py:1404`

固定サイズのスロット（ページ）に分割し、`deque[TensorMemoryObj]`でフリーリストを管理。

- **初期化**: `buffer`を`align_bytes`（=1チャンク分のKVデータサイズ）で分割
- **allocate()**: `free_blocks.popleft()` — O(1)
- **free()**: `free_blocks.append(mem_obj)` — O(1)、`invalidate()`しない（再利用）
- **スレッドセーフ**: `deque`のCPython実装がアトミックなため、ロック不要
- **用途**: P2P/PD共有時のNIXL連携（ページ単位のアドレス管理が必要）

## MixedMemoryAllocator — 通常時のデフォルト

**参照**: `target/LMCache/lmcache/v1/memory_management.py:1892`

LocalCPUBackendの`initialize_allocator()`がデフォルトで生成するアロケータ。

```
MixedMemoryAllocator
├── pin_allocator: TensorMemoryAllocator or PagedTensorMemoryAllocator
│   └── buffer: pinned CPU memory（NUMA-aware確保可能）
└── buffer_allocator: BufferAllocator
    └── 個別bytearrayを都度確保
```

- **MemoryFormat分岐**: KV_2LTD/KV_T2D/KV_2TD/KV_MLA_FMT → pin_allocator、BINARY_BUFFER → buffer_allocator
- **pinned memory確保**: `lmc_ops.alloc_pinned_ptr()`（CUDAのcudaHostAlloc相当）
  - NUMA対応時: `lmc_ops.alloc_pinned_numa_ptr(size, numa_id)`
- **close()**: `lmc_ops.free_pinned_ptr()`で明示解放

## MemoryObj — メモリオブジェクトの抽象

### TensorMemoryObj

**参照**: `target/LMCache/lmcache/v1/memory_management.py:431`

- **raw_data**: フラットな`torch.Tensor`（uint8ビュー）。アロケータのバッファスライス
- **metadata**: `MemoryObjMetadata`（shape, dtype, address, phy_size, ref_count, pin_count, fmt, cached_positions）
- **group_prefix_sum**: 複数グループ（shapes/dtypes）のバイトオフセットプレフィックスサム
- **tensor プロパティ**: `raw_data[:logical_size].view(dtype).view(shape)` でテンソルビューを返す
- **get_tensor(index)**: グループ別テンソルビュー（MLA等の複数形状対応）

### ref_count / pin_count ライフサイクル

```
確保時: ref_count=1, pin_count=0
  ↓
hot_cacheに登録: ref_count_up() → ref_count=2
  ↓
GPU転送中: ref_count_up() → ref_count=3
  ↓
GPU転送完了: ref_count_down() → ref_count=2
  ↓
hot_cacheから追い出し: ref_count_down() → ref_count=1
  ↓
batched_put完了: ref_count_down() → ref_count=0 → allocator.free()
```

**pin_count**: lookup時にpin()でEviction対象外にマーク。
- `can_evict = not is_pinned and ref_count == 1`（ref_count=1=hot_cacheのみが保持）
- PinMonitor: タイムアウト追跡。異常なpin長期化を検出

### BytesBufferMemoryObj

`bytes`/`bytearray`のラッパー。ref_count操作はno-op（GC任せ）。
BINARY_BUFFER形式用。Serde圧縮結果の格納に使用。

## メモリサイズ計算

### calculate_chunk_budget()

**参照**: `target/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py:688`

```python
max_chunks = total_memory // aligned_chunk_bytes
```

非同期ローディングシステムでの同時確保数上限を算出。
デッドロック防止のためのチャンクバジェット管理に使用。

### get_full_chunk_size()

Layerwise時: `chunk_tokens * kv_size * hidden_dim * dtype_size`（1レイヤー分）
Bulk時: `kv_size * num_layers * chunk_tokens * hidden_dim * dtype_size`（全レイヤー分）
