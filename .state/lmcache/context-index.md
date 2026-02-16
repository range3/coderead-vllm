# コンテキスト索引

各ドキュメントの内容を1行で要約。セッション開始時に読み、必要なドキュメントだけを選択的に読み込む。

## architecture/
- **overview.md** — LMCache全体アーキテクチャ、パッケージ構造（v1がメイン）、6エントリポイント、2動作モード、データフロー概要 [SHALLOW]
- **data-flow.md** — KVキャッシュ store + retrieve パスのエンドツーエンドフロー。store: adapter→Engine→TokenDB→GPUConnector→StorageManager。retrieve: Scheduler lookup→Worker start_load_kv→Engine→StorageManager→GPUConnector。Bulk/Layerwise 2モード、非同期prefetch、パイプラインGenerator設計。Mermaid図付き [MEDIUM] [VERIFIED]

## components/
- **vllm-integration/summary.md** — LMCacheConnectorV1Dynamic/Impl。Store: save_kv_layer()のレイヤーワイズGenerator駆動。Retrieve: get_num_new_matched_tokens(Scheduler lookup)→start_load_kv(Worker load)→wait_for_layer_load。LoadSpec/SaveSpec/ConnectorMetadata。LookupClient-LookupServer ZMQ IPC分離 [MEDIUM] [VERIFIED]
- **cache-engine/summary.md** — LMCacheEngine。Store: store_layer(Generator)/store(bulk)。Retrieve: retrieve(bulk)/retrieve_layer(layerwise Generator)/lookup。_process_tokens_internal(同期)/_async_process_tokens_internal(prefetch消費) [MEDIUM] [VERIFIED]
- **token-database/summary.md** — ChunkedTokenDatabase。256トークンチャンク分割、vLLM sha256_cbor完全互換ハッシュ、NONE_HASH、CacheEngineKey/LayerCacheEngineKey生成。Retrieve時もprocess_tokens()で同一ロジック。LookupClientも独自インスタンスを保持 [MEDIUM] [VERIFIED]
- **gpu-connector/summary.md** — Store: batched_from_gpu() Generator（Paged GPU→中間バッファ→Pinned CPU）。Retrieve: batched_to_gpu()。Bulk=multi_layer_kv_transfer一括。Layerwise=3段パイプライン（Load/RoPE補正/Paged書き込み）、ダブルバッファ、fused_rotary_emb [MEDIUM] [VERIFIED]
- **storage-manager/summary.md** — StorageManager: batched_put(全バックエンド配布)/batched_get(location指定+write-back)/layerwise_batched_get(非同期Future)/get_block_mapping(prefix match)。LocalCPUBackend: hot_cache同期R/W。async_lookup_and_prefetch(EventManager連携) [MEDIUM] [VERIFIED]

## glossary.md
- 用語集: CacheEngineKey, LayerCacheEngineKey, MemoryObj, MemoryFormat, TokenDatabase, StorageManager, GPUConnector, NONE_HASH, store_mask, slot_mapping, hot_cache, LookupClient, LookupServer, EventManager, token_mask, ret_mask, write-back, get_block_mapping, load_stream, fused_rotary_emb等。設定キー一覧あり [VERIFIED]
