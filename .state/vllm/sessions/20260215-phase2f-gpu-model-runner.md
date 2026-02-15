# Phase 2f: GPUModelRunner深堀り [SHALLOW]→[MEDIUM]

**日付**: 2026-02-15
**目的**: GPUModelRunnerのKVCache-GPU接続パス、InputBatch永続バッチ、CUDAGraph統合を調査

## 成果物

1. `docs/src/components/gpu-model-runner/kv-cache-interface.md` — 新規作成
2. `docs/src/components/gpu-model-runner/input-batch.md` — 新規作成
3. `docs/src/components/gpu-model-runner/summary.md` — [SHALLOW]→[MEDIUM] 昇格

## 主要な発見

### KVCache-GPU Interface (4段変換パス)

1. **ブロックID取込** (`_update_states()` L874): 3ケース — 新規(add_request)、継続(extend+append_row)、プリエンプション復帰(全置換+add_request)
2. **スロットマッピング計算** (`BlockTable.compute_slot_mapping()` L133): `block_number * block_size + (position % block_size)` の変換式。Hybrid Blockは`map_to_kernel_blocks()`で割当ブロック→カーネルブロックに展開
3. **DMA転送** (`_prepare_inputs()` L1454): `commit_block_table()` を最初に呼びDMAとCPU演算をオーバーラップ
4. **2形式出力** (`_get_slot_mappings()` L3237): by_gid (AttentionMetadata用) / by_layer (ForwardContext→reshape_and_cache用)
5. **AttentionMetadata構築** (`_build_attention_metadata()` L1673): CommonAttentionMetadata→shallow copy→per-group block_table差替→per-layer AttentionMetadata (builder.build())

### InputBatch永続バッチ

- `CachedRequestState` (L30): リクエスト論理状態。プリエンプション後もself.requestsに保持
- `InputBatch` (L81): 事前割り当てCPUテンソル群。`token_ids_cpu_tensor` は `(max_num_reqs, max_model_len)` で大きくなりうる
- `MultiGroupBlockTable` (L253): KVキャッシュグループごとにBlockTableを保持
- `CpuGpuBuffer` (L105): ピン留めCPU + GPU ペア、numpy view公開、non-blocking DMA
- `condense()` (L626): 空スロットを圧縮して連続配列を保つ（末尾リクエストを前方空きスロットに移動）

### CUDAGraph統合

- **3モード**: FULL（forward全体キャプチャ）、PIECEWISE（Attention以外、torch.compile）、NONE（Eager）
- **CudagraphDispatcher** (L14): cudagraph_keys辞書でキャプチャ済みバッチ記述子を管理。dispatch()はFULL→PIECEWISE→NONEの順に探索
- **パディング**: 実際のトークン数を最小キャプチャサイズに丸め上げ。未使用スロット=-1 (PAD_SLOT_ID)
- **uniform_decode**: 全リクエストがdecodeフェーズ(query_len=1)の判定。FULL CUDAGraphの主要使用条件

## 読み込んだファイル

- `gpu_model_runner.py`: L874-1113 (_update_states), L1454-1672 (_prepare_inputs), L1673-1915 (_build_attention_metadata), L3076-3199 (_determine_batch_execution_and_padding), L3237-3310 (_get_slot_mappings), L3380-3544 (execute_model内の呼び出し順序)
- `block_table.py`: 全体 (343行) — BlockTable, compute_slot_mapping, map_to_kernel_blocks, MultiGroupBlockTable
- `gpu_input_batch.py`: L30-79 (CachedRequestState), L81-270 (InputBatch.__init__), L304-420 (add_request), L469-521 (remove_request), L626-705 (condense)
- `cudagraph_dispatcher.py`: L1-120 (初期化), L220-289 (dispatch)
- `utils.py`: L105-141 (CpuGpuBuffer)

## 新たな疑問

- ForwardContextでのslot_mappings_by_layer消費: reshape_and_cache()の具体的な実装場所は？
- CUDAGraph FULL mode時にslot_mappingsは毎step書き直されるか、それとも固定か？
