# Phase 2g: KV Transfer / LMCache 調査

**日付**: 2026-02-15
**目的**: KV Transfer / LMCache のコンポーネント文書化（[SHALLOW]→[MEDIUM]昇格）

## 成果物

1. `docs/src/components/kv-transfer/summary.md` — KV Transfer コンポーネント文書 [MEDIUM]
2. `docs/src/investigations/lmcache-integration.md` — LMCache統合調査報告 [MEDIUM]
3. `docs/src/glossary.md` — 10用語追加（KVConnectorBase_V1, KVConnectorFactory, KVTransferConfig, KVConnectorModelRunnerMixin, Disaggregated Prefill, WAITING_FOR_REMOTE_KVS, KV Cache Events, CacheEngineKey等）

## 主要な発見

### vLLM側 KV Transfer

- **KVConnectorBase_V1**: 7 abstractメソッド（Worker側4+Scheduler側3）、2ロール分離
- **KVConnectorFactory**: 10個の登録済みコネクタ、遅延ロード、動的ロード対応
- **Scheduler統合**: get_num_new_matched_tokens()→allocate_slots()→update_state_after_alloc()→build_connector_meta()のフロー。非同期ロード時はWAITING_FOR_REMOTE_KVS状態
- **Worker統合**: KVConnectorModelRunnerMixin._get_kv_connector_output()コンテキストマネージャでライフサイクル管理
- **KV Cache Events**: BlockStored/BlockRemoved/AllBlocksCleared 3イベント型、ZmqEventPublisher配信
- **cross-layer blocks**: prefer_cross_layer_blocks=Trueで全レイヤーKVを連続テンソルに配置

### LMCache側

- **チャンク単位保存**: デフォルト256トークン/チャンク、CacheEngineKey（プレフィックスハッシュ）
- **3層ストレージ**: LocalCPU(L1) → LocalDisk(L2) → Remote(L3)
- **15+リモートコネクタ**: Redis, S3, FS, Mooncake, Valkey等
- **vLLMアダプタ**: native（vLLM同梱）/ latest（LMCacheパッケージ）の2パス分岐
- **LMCacheConnectorV1Impl**: Scheduler側=LookupClient、Worker側=LMCacheEngine+LookupServer+ZMQOffloadServer
- **RequestTracker**: リクエスト状態追跡。from_new_request()→update()のライフサイクル
- **ReqMeta**: slot_mapping, LoadSpec, SaveSpecでロード/セーブ仕様を伝達
- **GPUConnector 3種**: Paged/PagedLayerwise/BufferLayerwise
- **セーブ判定**: チャンク境界到達・デコードフェーズ・Disagg状態で分岐

## 読んだファイル

| ファイル | 行 | 内容 |
|---------|---|------|
| `target/vllm/.../kv_connector/v1/base.py` | 全体 | KVConnectorBase_V1 |
| `target/vllm/.../kv_connector/factory.py` | 全体 | KVConnectorFactory |
| `target/vllm/.../kv_transfer_state.py` | 全体 | グローバル状態管理 |
| `target/vllm/.../config/kv_transfer.py` | 全体 | KVTransferConfig |
| `target/vllm/.../kv_events.py` | 全体 | KV Cache Events |
| `target/vllm/.../kv_connector_model_runner_mixin.py` | 全体 | Mixin |
| `target/vllm/.../lmcache_connector.py` | 全体 | LMCacheラッパー |
| `target/vllm/.../lmcache_integration/vllm_v1_adapter.py` | L1-843 | native実装 |
| `target/LMCache/lmcache/v1/cache_engine.py` | L1-100 | LMCacheEngine |
| `target/vllm/.../scheduler.py` | L600-770, L1920-2040 | Scheduler KV統合 |
| `target/vllm/.../gpu_model_runner.py` | KVコネクタ関連箇所 | GPUModelRunner統合 |

## 残課題

- NixlConnector詳細（RDMA、pre-register、handshake）
- OffloadingConnector詳細
- LMCache CacheBlend動作
- cross-layer blocksの性能影響
- LookupClient/Serverの通信プロトコル
