# Phase 1 セッション1: Store パス垂直スライス

**日付**: 2026-02-16
**Phase**: 1（セッション1/2）
**対象OSS**: LMCache

## 目標

vLLM統合でのKVキャッシュ store パス（GPU→CPU→Storage）を入力から出力まで完全追跡。

## 追跡したパス

```
vLLM attention layer (save_kv_layer hook)
  → LMCacheConnectorV1Dynamic.save_kv_layer()     [lmcache_connector_v1.py:90]
  → LMCacheConnectorV1Impl.save_kv_layer()        [vllm_v1_adapter.py:964]
  → LMCacheEngine.store_layer()                    [cache_engine.py:528] (generator)
    → ChunkedTokenDatabase.process_tokens()        [token_database.py:309]
    → StorageManager.batched_allocate()             [storage_manager.py:352]
    → GPUConnector.batched_from_gpu()              [gpu_connectors.py:1212] (generator)
      → lmc_ops.single_layer_kv_transfer()         (CUDA kernel)
    → StorageManager.batched_put()                  [storage_manager.py:388]
      → LocalCPUBackend.submit_put_task()          [local_cpu_backend.py:141]
```

## 主要な発見

1. **2つの入れ子Generator**: store_layer()とbatched_from_gpu()が入れ子Generator構成。adapterが各attentionレイヤー後にnext()で進める
2. **ハッシュ完全互換**: TokenDatabaseはvLLMのsha256_cborハッシュ関数を直接import。NONE_HASHもvLLMから取得。完全に同一のアルゴリズム
3. **LMCacheManagerにstore()なし**: adapterがEngineを直接呼ぶ。Managerはライフサイクル管理のみ
4. **同時配布（非プロモーション）**: StorageManager.batched_put()は全バックエンドに同時書き込み。L1→L2プロモーションではない
5. **LocalCPUBackend同期実行**: hot_cacheへの書き込みはバックグラウンドスレッドなし、cpu_lock下で即座に完了
6. **CUDAストリーム分離**: store_streamを使用しメイン計算とオーバーラップ。最初のリクエストのみ同期
7. **LayerCacheEngineKey**: CacheEngineKeyにlayer_idを付与してレイヤー単位保存。layer 0のキーでcontains()判定

## 作成ドキュメント

- `docs/src/lmcache/architecture/data-flow.md` — storeパスフロー（Mermaid図付き）
- `docs/src/lmcache/components/vllm-integration/summary.md`
- `docs/src/lmcache/components/cache-engine/summary.md`
- `docs/src/lmcache/components/token-database/summary.md`
- `docs/src/lmcache/components/gpu-connector/summary.md`
- `docs/src/lmcache/components/storage-manager/summary.md`
- `docs/src/lmcache/glossary.md` 更新（LayerCacheEngineKey, NONE_HASH, store_mask, slot_mapping, hot_cache等追加）

## 解決した疑問

- LMCacheEngine store処理のタイミング → save_kv_layer()フックからLayer 0でGenerator生成
- chunk_hashアルゴリズムのvLLM互換性 → 完全同一（sha256_cbor直接利用）
- from_gpu slot_mappingの出自 → ConnectorMetadata.requests[i].slot_mapping
- バックエンド間データ移動 → 同時配布（プロモーションではない）

## 次回の作業

Phase 1 セッション2: retrieve パス（Storage→CPU→GPU）を追跡。
