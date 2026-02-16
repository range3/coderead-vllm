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

## 主要メソッド（Retrieve方向）

### Scheduler側: get_num_new_matched_tokens()

**参照**: `target/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py:1193`

vLLM Schedulerの`schedule()`から呼ばれ、外部KVキャッシュのヒット数を返す。

1. `LookupClient.lookup_cache(req_id)`で既存キャッシュ確認（2回目以降）
2. 未キャッシュなら`LookupClient.lookup(token_ids, req_id)`でZMQ経由でWorker側に問い合わせ
3. `LoadSpec(vllm_cached_tokens, lmcache_cached_tokens, can_load=False)`を生成
4. `update_state_after_alloc()`で`can_load=True`に更新（ブロック確保成功時）
5. `build_connector_meta()`で`ReqMeta(load_spec=LoadSpec)`を`LMCacheConnectorMetadata`に格納

### Worker側: start_load_kv()

**参照**: `target/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py:737`

vLLMのforward実行**前**に`ForwardContext`から呼ばれ、KVキャッシュのGPU復元を開始。

**token_maskの構築**:
1. `request.load_spec.vllm_cached_tokens`をchunk_sizeの倍数に切り下げ → `masked_token_count`
2. `token_mask[:masked_token_count] = False`（vLLM既キャッシュ分）、残り=True

**2モード分岐**:
- **Layerwise** (`use_layerwise=True`):
  1. `LMCacheEngine.retrieve_layer()`でGenerator取得
  2. `next()` × 2回で先行2レイヤー分をキック
  3. `self.layerwise_retrievers`にGenerator追加
- **Bulk** (`use_layerwise=False`):
  1. `LMCacheEngine.retrieve()`を呼び出し、`ret_mask`を取得
  2. 取得失敗時は`record_failed_blocks()`で失敗ブロックIDを`_invalid_block_ids`に記録

### Worker側: wait_for_layer_load()

**参照**: `target/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py:940`

各attentionレイヤー実行**前**に呼ばれ、該当レイヤーのKVロード完了を待機。
`layerwise_retrievers`内の全Generatorを`next()`で1ステップ進行。
最終レイヤーでは`ret_mask`を取得して検証。

## 設計上の注意点

- `LMCacheConnectorV1Dynamic`は純粋な委譲シェル。全メソッドが`self._lmcache_engine`（V1Impl）に転送
- **LMCacheManagerにstore()メソッドは存在しない**。V1ImplがEngineを直接呼ぶ
- `kv_role`は`"kv_both"`（default）/`"kv_producer"`/`"kv_consumer"`の3値。producer時はskip_leading_tokens=0（全トークンstore）
- `current_layer`カウンタでレイヤー追跡。wait_for_save()でリセット
- **Retrieve 2モード**: `use_layerwise`（デフォルトFalse）でBulk/Layerwiseを切替
- **LookupClient-LookupServer分離**: SchedulerプロセスのLookupClientからWorkerプロセスのLookupServerにZMQ IPC通信
- **LoadSpec**: Scheduler→Worker間でlookup結果を伝達するデータ構造（ConnectorMetadata経由）
