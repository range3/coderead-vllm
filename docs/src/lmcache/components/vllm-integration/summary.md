# vLLM統合（LMCacheConnector）

> **深度**: [MEDIUM] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 1 セッション1）

## 概要

LMCacheとvLLMを接続するアダプタ層。vLLMの`KVConnectorBase_V1`インターフェースを実装し、
attentionレイヤー実行中のKVキャッシュstore/retrieveをLMCacheに委譲する。

## クラス階層

```
KVConnectorBase_V1 (vLLM)
  └── LMCacheConnectorV1Dynamic       ← vLLMに登録される外殻
        └── LMCacheConnectorV1Impl    ← 実装本体（Worker側）
              └── LMCacheEngine       ← 直接参照（LMCacheManager経由で取得）
```

**参照**:
- `target/LMCache/lmcache/integration/vllm/lmcache_connector_v1.py` (Dynamic)
- `target/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py:964` (Impl.save_kv_layer)

## 主要メソッド（Store方向）

### save_kv_layer()

**シグネチャ**: `save_kv_layer(layer_name: str, kv_layer: Tensor, attn_metadata: AttentionMetadata, **kwargs)`

vLLMの各attentionレイヤー実行後に呼ばれる。

- `use_layerwise=False`の場合は即座にreturn（非レイヤーワイズパスは別経路）
- `kv_role="kv_consumer"`の場合もreturn（consumeのみ、storeしない）

**Layer 0の処理**:
1. `connector_metadata.requests`から`save_spec.can_save=True`のリクエストを抽出
2. `skip_leading_tokens`をchunk_size（256）の倍数に切り下げてマスク整合
3. `store_mask`を構築：prefix部分=False、新規部分=True
4. `LMCacheEngine.store_layer()`でGenerator生成、`self.layerwise_storers`に追加
5. 最初のリクエストのみ`sync=True`（CUDAストリーム同期）

**全レイヤー共通**: `layerwise_storers`内の全Generatorを`next()`で1ステップ進行。

### ConnectorMetadata

**参照**: `target/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` (LMCacheConnectorMetadata)

Scheduler側で構築され、Worker側に渡される。各リクエストの`token_ids`、`slot_mapping`、`save_spec`（`can_save`, `skip_leading_tokens`）を含む。

## 上流・下流

- **上流**: vLLM GPUModelRunner（`save_kv_layer`フック）
- **下流**: LMCacheEngine（`store_layer` / `store` / `retrieve` / `retrieve_layer`）
- **関連**: LMCacheManager（ライフサイクル管理、Engine取得）

## 設計上の注意点

- `LMCacheConnectorV1Dynamic`は純粋な委譲シェル。全メソッドが`self._lmcache_engine`（V1Impl）に転送
- **LMCacheManagerにstore()メソッドは存在しない**。V1ImplがEngineを直接呼ぶ
- `kv_role`は`"kv_both"`（default）/`"kv_producer"`/`"kv_consumer"`の3値。producer時はskip_leading_tokens=0（全トークンstore）
- `current_layer`カウンタでレイヤー追跡。wait_for_save()でリセット
