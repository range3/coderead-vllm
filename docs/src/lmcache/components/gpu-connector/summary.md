# GPUConnector

> **深度**: [MEDIUM] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 1 セッション1）

## 概要

GPU上のKVキャッシュとCPU上のMemoryObj間でデータを転送するコンポーネント。
vLLMのページドメモリレイアウトからslot_mappingを使って正しいデータを抽出する。

**参照**: `target/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py`

## クラス階層

```
GPUConnectorInterface (ABC)
  ├── VLLMPagedMemGPUConnectorV2        ← 非レイヤーワイズ（全レイヤー一括）
  ├── VLLMPagedMemLayerwiseGPUConnector ← レイヤーワイズ（主要パス）
  ├── VLLMBufferLayerwiseGPUConnector   ← CacheBlend用（中間バッファ経由）
  ├── VLLMGPUConnectorXPU               ← Intel XPU用
  └── SGLangGPUConnector                ← SGLang用
```

## 主要メソッド（Store方向）

### batched_from_gpu()（VLLMPagedMemLayerwiseGPUConnector）

**参照**: `target/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py:1212`

```python
def batched_from_gpu(
    memory_objs: List[List[MemoryObj]],  # [num_layers][num_chunks]
    starts: List[int],
    ends: List[int],
    **kwargs,  # slot_mapping, sync, kvcaches
) -> Generator
```

**Generator関数**。`num_layers + 1`回yield。

**セットアップフェーズ**:
1. slot_mapping_chunksを結合して`slot_mapping_full`を構築
2. `use_gpu=True`時: `gpu_buffer_allocator`から中間GPUバッファを確保

**レイヤーループ（各yield間）**:

| ステップ | use_gpu=True | use_gpu=False |
|---|---|---|
| 1 | `lmc_ops.single_layer_kv_transfer()`<br/>paged GPU → 中間GPUバッファ | `lmc_ops.single_layer_kv_transfer()`<br/>paged GPU → 直接pinned CPU |
| 2 | `memory_obj.tensor.copy_(..., non_blocking=True)`<br/>GPUバッファ → pinned CPU | （不要） |

**lmc_ops.single_layer_kv_transfer引数**:
```python
lmc_ops.single_layer_kv_transfer(
    dst_tensor,          # 出力先
    kvcaches[layer_id],  # vLLMのページドKVキャッシュ（1レイヤー分）
    slot_mapping,        # トークン位置→flat slot
    True,                # store=True（GPU→dst方向）
    True,                # token_major=True（KV_T2D形式）
    vllm_two_major,      # vLLMの2-major形式フラグ
    use_mla,             # MLA形式フラグ
)
```

**CUDAストリーム設計**:
- `self.store_stream`: 専用CUDAストリーム。メイン計算ストリームとオーバーラップ可能
- `store_stream.wait_stream(current_stream)`: 計算が完了してからDMA開始
- `sync=True`時のみ`store_stream.synchronize()`で同期（最初のリクエストのみ）

**出力形式**:
- 標準: `MemoryFormat.KV_T2D` = `[num_tokens, 2, hidden_dim]`
- MLA: `MemoryFormat.KV_MLA_FMT` = `[num_tokens, hidden_dim]`

### get_shape()

**参照**: `target/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py:1331`

```python
def get_shape(num_tokens: int) -> torch.Size:
    # 標準: [num_tokens, 2, hidden_dim_size]
    # MLA:  [num_tokens, hidden_dim_size]
```

## 主要メソッド（Retrieve方向）

### batched_to_gpu()（VLLMPagedMemGPUConnectorV2 — Bulk）

**参照**: `target/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py:359`

```python
def batched_to_gpu(memory_objs, starts, ends, **kwargs)
```

`load_stream`上で全チャンクの`to_gpu()`を順次実行し、最後に`load_stream.synchronize()`。
`to_gpu()`は`lmc_ops.multi_layer_kv_transfer()`でMemoryObj → pagedメモリに**全レイヤー一括**転送。

### batched_to_gpu()（VLLMBufferLayerwiseGPUConnector — Layerwise）

**参照**: `target/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py:683`

**Generator関数**。`num_layers + 2`回イテレーションの**3段パイプライン**:

| 段 | 操作 | ストリーム |
|---|---|---|
| Load | CPU pinned → GPUバッファに`copy_(non_blocking=True)` | `load_stream` |
| Compute | RoPE位置補正 + gap zeroing | default stream |
| Write | `lmc_ops.single_layer_kv_transfer()`でバッファ→pagedメモリ | default stream |

**ダブルバッファ**: `compute_gpu_buffer_obj`と`load_gpu_buffer_obj`をping-pongし、DMAとRoPE計算をオーバーラップ。

**RoPE位置補正**（`cache_positions=True`時）:
- `MemoryObjMetadata.cached_positions`から保存時位置を取得
- `fused_rotary_emb(old_positions, new_positions, K_tensor)`で差分補正
- これにより異なるコンテキスト位置でもKVキャッシュを再利用可能

**gap zeroing**: 連続しないチャンク間のギャップ位置をゼロ埋め。

**`mem_obj_consumer.send()`パターン**: Engine側から`send(mem_objs_layer)`でデータを受け取る（Generator.send）。`yield`でデータ受領→次イテレーションで処理。

## 上流・下流

- **上流**: LMCacheEngine（store_layer/store/retrieve等で呼び出し）
- **下流**: vLLMのページドKVキャッシュ（`self.kvcaches`）、lmc_ops CUDAカーネル
- **依存**: lmcache.c_ops（single_layer_kv_transfer / multi_layer_kv_transfer）

## 設計上の注意点

- 中間GPUバッファ（`use_gpu=True`）は**全チャンクを結合してから一括転送**するため、チャンクごとのカーネル起動オーバーヘッドを削減
- `kvcaches`はvLLMの`kv_cache`リスト（`list[Tensor]`、レイヤーごとに1テンソル）を`initialize_kvcaches_ptr()`で受け取る
- GPUバッファは`gpu_buffer_allocator`から確保し、使用後に`ref_count_down()`で解放
- **Retrieve時のCUDAカーネル**: Bulk=`multi_layer_kv_transfer`（全レイヤー一括）、Layerwise=`single_layer_kv_transfer`（store=Falseで逆方向=メモリ→paged）
