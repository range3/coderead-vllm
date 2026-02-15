# コンテキスト索引

各ドキュメントの内容を1行で要約。セッション開始時に読み、必要なドキュメントだけを選択的に読み込む。

## architecture/
- **overview.md** — LMCache全体アーキテクチャ、パッケージ構造（v1がメイン）、6エントリポイント、2動作モード、データフロー概要 [SHALLOW]
- **data-flow.md** — KVキャッシュ store パスのエンドツーエンドフロー。adapter→Engine→TokenDB→GPUConnector→StorageManager→LocalCPUBackend。パイプラインGenerator設計。Mermaid図付き [MEDIUM] [VERIFIED]

## components/
- **vllm-integration/summary.md** — LMCacheConnectorV1Dynamic/Impl。vLLM KVConnectorBase_V1実装、save_kv_layer()のレイヤーワイズGenerator駆動、ConnectorMetadata/SaveSpec [MEDIUM] [VERIFIED]
- **cache-engine/summary.md** — LMCacheEngine。store_layer()(Generator) / store()(bulk) の2 API。TokenDB+GPUConnector+StorageManager統合 [MEDIUM] [VERIFIED]
- **token-database/summary.md** — ChunkedTokenDatabase。256トークンチャンク分割、vLLM sha256_cbor完全互換ハッシュ、NONE_HASH、CacheEngineKey/LayerCacheEngineKey生成 [MEDIUM] [VERIFIED]
- **gpu-connector/summary.md** — VLLMPagedMemLayerwiseGPUConnector。batched_from_gpu() Generator、lmc_ops.single_layer_kv_transfer CUDAカーネル、store_stream、KV_T2D形式 [MEDIUM] [VERIFIED]
- **storage-manager/summary.md** — StorageManager（多段ディスパッチ）+ LocalCPUBackend（L1 hot_cache同期書き込み）。batched_put/batched_allocate、CachePolicy、ref_count管理 [MEDIUM] [VERIFIED]

## glossary.md
- 用語集: CacheEngineKey, LayerCacheEngineKey, MemoryObj, MemoryFormat, TokenDatabase, StorageManager, GPUConnector, NONE_HASH, store_mask, slot_mapping, hot_cache等。設定キー一覧あり [VERIFIED]
