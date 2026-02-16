# Phase 1 セッション2: Retrieve パス垂直スライス

**日付**: 2026-02-16
**Phase**: 1（セッション2/2）
**対象**: LMCache

## 目標

vLLM統合でのKVキャッシュ retrieve パスを入力から出力まで完全に追跡する。

## 追跡したパス

### Scheduler側（lookup）
1. `V1Impl.get_num_new_matched_tokens()` — vLLM Schedulerから呼ばれる
2. `LookupClient.lookup(token_ids, req_id)` — ZMQ IPCでWorker側に問い合わせ
3. `LoadSpec(vllm_cached, lmcache_cached, can_load)` — ヒット結果を構造化
4. `update_state_after_alloc()` → `can_load=True`
5. `build_connector_meta()` → `ReqMeta(load_spec)` を `ConnectorMetadata` に格納

### Worker側（load — Bulkモード）
1. `V1Impl.start_load_kv(forward_context)` — Forward前に呼ばれる
2. `token_mask` 構築: vLLM cached分をFalse（chunk_size倍数に切り下げ）
3. `LMCacheEngine.retrieve(tokens, mask, kvcaches, slot_mapping, ...)`
4. `_process_tokens_internal()` / `_async_process_tokens_internal()`
5. `StorageManager.get_block_mapping()` → prefix matchでバックエンド特定
6. `StorageManager.batched_get(keys, location)` → `LocalCPUBackend.batched_get_blocking()` → `MemoryObj`取得
7. `GPUConnector.batched_to_gpu(memory_objs, starts, ends, slot_mapping)` → `lmc_ops.multi_layer_kv_transfer()`
8. `memory_obj.ref_count_down()` で解放
9. `ret_mask` 返却

### Worker側（load — Layerwiseモード）
1. `LMCacheEngine.retrieve_layer(tokens, mask, ...)` — Generator
2. `StorageManager.layerwise_batched_get(keys_layer_major)` → 非同期Future Generator
3. `GPUConnector.batched_to_gpu(starts, ends)` → mem_obj_consumer Generator
4. レイヤーごと: `task.result()` → `mem_obj_consumer.send(mem_objs)` → 3段パイプライン（Load/RoPE/Write）
5. `wait_for_layer_load()` で各レイヤー同期

### 非同期prefetchパス
1. `LookupClient.lookup()` → ZMQ → `LookupServer`
2. `StorageManager.async_lookup_and_prefetch()` — 全バックエンドにprefix matchで`batched_get_non_blocking()`
3. `EventManager.add_event(LOADING, lookup_id, all_done_task)`
4. Worker側: `EventManager.pop_event(LOADING, req_id)` → `future.result()` → `_async_process_tokens_internal()`

## 主要な発見

1. **Retrieve 2モード**: Bulk（デフォルト、全レイヤー一括）と Layerwise（Generator、パイプライン可能）
2. **Scheduler-Worker分離**: LookupClientがZMQ IPCでScheduler→Worker通信。LoadSpecでヒット情報伝達
3. **非同期prefetch**: lookup時にprefetchをトリガーし、EventManagerでretrieveと紐付け
4. **StorageManager write-back**: リモートバックエンドから取得→自動的にLocalCPUBackendにコピー
5. **get_block_mapping prefix match**: 各バックエンドのbatched_contains()で連続ヒット数を取得し、残りを次バックエンドに渡す
6. **Layerwise 3段パイプライン**: Load(CPU→GPU DMA) / Compute(RoPE補正+gap zeroing) / Write(バッファ→paged)、ダブルバッファでオーバーラップ
7. **RoPE位置補正**: MemoryObjMetadata.cached_positionsから保存時位置を取得し、fused_rotary_embで差分補正
8. **start_load_kvで先行2レイヤーキック**: Layerwise時にnext()×2で先行2レイヤー分をキック
9. **token_maskのchunk_size切り下げ**: vLLM cached tokensをchunk_sizeの倍数に切り下げ、オーバーラップ領域はLMCacheが上書き
10. **部分取得失敗→_invalid_block_ids**: record_failed_blocks()で失敗ブロックIDを計算、vLLM Schedulerが回収して再計算指示

## 解決した疑問

- retrieve時のprefetchはScheduler側lookupがトリガー、EventManagerで紐付け
- to_gpu/from_gpuのslot_mappingは同一経路（ConnectorMetadata.requests[i].slot_mapping）
- LookupClient-LookupServer間はZMQ IPC（REQ/REP）、hashes+offsetsをmsgpack送信
- MemoryObjのライフサイクル: batched_get_non_blockingでref_count_up → retrieve完了後ref_count_down

## 更新したドキュメント

- `docs/src/lmcache/architecture/data-flow.md` — retrieveパス全体を追記
- `docs/src/lmcache/components/vllm-integration/summary.md` — Scheduler/Worker側retrieveメソッド追記
- `docs/src/lmcache/components/cache-engine/summary.md` — retrieve()/retrieve_layer()/lookup() API追記
- `docs/src/lmcache/components/storage-manager/summary.md` — batched_get/layerwise_batched_get/get_block_mapping/async_lookup_and_prefetch追記
- `docs/src/lmcache/components/gpu-connector/summary.md` — batched_to_gpu (Bulk/Layerwise)追記
- `docs/src/lmcache/components/token-database/summary.md` — Retrieve時の利用追記
- `docs/src/lmcache/glossary.md` — LookupServer, EventManager, token_mask, ret_mask, write-back等追加
