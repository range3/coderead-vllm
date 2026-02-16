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
- **storage-manager/summary.md** — StorageManager多段バックエンド管理 + LocalCPUBackendのAllocator/Cache二重役割。バックエンド階層(L1/L2/L3)、batched_put全配布、write-back、freeze/bypass、独自バックエンドインターフェース。サブドキュメント3本 [DEEP] [VERIFIED]
- **storage-manager/memory-allocator.md** — アロケータ階層（TensorMemoryAllocator explicit free list、PagedTensorMemoryAllocator固定スロット、MixedMemoryAllocator）、AddressManager(SortedList+4KB align+coalesce)、MemoryObj/TensorMemoryObj ref_count/pin_countライフサイクル [DEEP] [VERIFIED]
- **storage-manager/cache-policy.md** — BaseCachePolicy 4メソッド、FIFO(dict先頭)、LRU(OrderedDict+move_to_end+再利用時間追跡)、LFU(SortedDict freq管理)、MRU(OrderedDict末尾)。Eviction発動フロー、can_evict条件(not pinned && ref_count==1) [DEEP] [VERIFIED]
- **storage-manager/local-disk-backend.md** — LocalDiskBackend L2永続化。AsyncPQThreadPoolExecutor(優先度:prefetch>delete>put)、O_DIRECT対応、容量Eviction、DiskCacheMetadata、独自バックエンドStoragePluginInterface実装ガイド(必須/オプション全メソッド一覧) [DEEP] [VERIFIED]

## glossary.md
- 用語集: CacheEngineKey, LayerCacheEngineKey, MemoryObj, MemoryFormat, TokenDatabase, StorageManager, GPUConnector, NONE_HASH, store_mask, slot_mapping, hot_cache, LookupClient, LookupServer, EventManager, token_mask, ret_mask, write-back, get_block_mapping, load_stream, fused_rotary_emb等。設定キー一覧あり [VERIFIED]
