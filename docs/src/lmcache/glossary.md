# LMCache 用語集

> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 1 セッション1）

## コアコンセプト

| 用語 | 説明 |
|------|------|
| **LMCacheEngine** | KVキャッシュのstore/retrieve/prefetchを統合するメインエンジン。`lmcache/v1/cache_engine.py` |
| **LMCacheManager** | LMCacheの内部コンポーネント（Engine, LookupClient, OffloadServer等）のライフサイクル管理。`lmcache/v1/manager.py` |
| **CacheEngineKey** | KVキャッシュチャンクの一意識別子。(model_name, world_size, worker_id, chunk_hash, dtype, request_configs)の6タプル。`lmcache/utils.py:333` |
| **LayerCacheEngineKey** | CacheEngineKey + layer_id。レイヤー単位保存時のキー。`split_layers()`で生成。`lmcache/utils.py:392` |
| **MemoryObj** | KVキャッシュデータを保持する抽象メモリオブジェクト。フォーマット情報とテンソルデータを包含。`lmcache/v1/memory_management.py` |
| **MemoryFormat** | KVキャッシュのメモリレイアウト種別。KV_2LTD, KV_T2D, KV_2TD, BINARY, KV_MLA_FMT等。 |
| **LMCacheMetadata** | モデル名、world_size、worker_id、kv_dtype、kv_shape等のメタ情報。サービングエンジンから抽出。 |
| **LMCacheEngineConfig** | YAML/環境変数ベースの設定。chunk_size, ストレージ設定, blend設定, P2P設定等。 |

## トークン処理

| 用語 | 説明 |
|------|------|
| **TokenDatabase** | トークン列をチャンクキー列に変換する抽象基底クラス。 |
| **ChunkedTokenDatabase** | 固定サイズ（default 256トークン）チャンクでプレフィックスハッシュを計算。標準の実装。 |
| **SegmentTokenDatabase** | セパレータベースでセグメント分割。CacheBlend時に使用。 |
| **chunk_size** | チャンクのトークン数。デフォルト256。 |
| **chunk_hash** | チャンクのプレフィックスハッシュ値。vLLMの`sha256_cbor`ハッシュ関数を直接利用（完全互換）。 |
| **NONE_HASH** | プレフィックスハッシュチェーンの初期値。vLLMの`kv_cache_utils.init_none_hash()`で初期化。 |
| **store_mask** | store時のマスク。False=already-cached prefix、True=新規トークン。False数はchunk_sizeの倍数必須。 |

## ストレージ

| 用語 | 説明 |
|------|------|
| **StorageManager** | 複数のストレージバックエンドを階層管理。put/get要求を各バックエンドに振り分け。 |
| **StorageBackendInterface** | ストレージバックエンドの抽象インターフェース。contains/put/get等。 |
| **LocalCPUBackend** | CPU メモリ上のKVキャッシュストレージ（L1）。hot_cache（OrderedDict）で管理。同期書き込み。 |
| **hot_cache** | LocalCPUBackendの`OrderedDict[CacheEngineKey, MemoryObj]`。CachePolicyでEviction管理。 |
| **allocator_backend** | MemoryObj確保を担当するバックエンド。通常はLocalCPUBackend。 |
| **LocalDiskBackend** | ディスク上のKVキャッシュストレージ（L2）。 |
| **RemoteBackend** | リモートストレージ（L3）。connector経由でRedis/S3/Valkey等に接続。 |
| **P2PBackend** | インスタンス間のP2P KVキャッシュ転送。 |
| **NIXLBackend** | NVIDIA NIXL経由の高速転送。 |
| **GdsBackend** | GPUDirect Storage経由の転送。 |
| **CachePolicy** | Eviction方針。FIFO/LRU/LFU/MRUから選択可能。 |
| **Serde** | シリアライゼーション/デシリアライゼーション。naive（無圧縮）、CacheGen（圧縮）、KIVI等。 |

## GPU連携

| 用語 | 説明 |
|------|------|
| **GPUConnectorInterface** | GPU KVキャッシュとCPU MemoryObj間のデータ転送抽象インターフェース。to_gpu/from_gpu。 |
| **VLLMPagedMemGPUConnectorV2** | vLLMのPaged KVキャッシュ向けGPUコネクタ（非レイヤーワイズ）。全レイヤー一括転送。 |
| **VLLMPagedMemLayerwiseGPUConnector** | レイヤー単位でKVキャッシュを転送するコネクタ。ジェネレータパターン使用。主要パス。 |
| **lmc_ops.single_layer_kv_transfer** | CUDAカーネル。vLLMのページドKVキャッシュからslot_mapping経由でデータを抽出/書き戻し。 |
| **slot_mapping** | トークン位置→vLLMページドメモリのflat slot位置へのマッピング。GPU Tensor。 |
| **store_stream** | GPU→CPU転送専用CUDAストリーム。メイン計算ストリームとオーバーラップ可能。 |
| **load_stream** | CPU→GPU転送専用CUDAストリーム。retrieve時にメイン計算とオーバーラップ。 |
| **lmc_ops.multi_layer_kv_transfer** | CUDAカーネル。全レイヤー一括でMemoryObj→paged KVキャッシュに転送（Bulk retrieve用）。 |
| **fused_rotary_emb** | RoPE位置補正関数。Layerwise retrieve時に保存時と現在のposition差分を補正。 |
| **VLLMBufferLayerwiseGPUConnector** | CacheBlend対応のLayerwiseコネクタ。ダブルバッファ+RoPE補正+gap zeroing。 |

## 統合

| 用語 | 説明 |
|------|------|
| **LMCacheConnectorV1Dynamic** | vLLMの`KVConnectorBase_V1`実装。`LMCacheConnectorV1Impl`に委譲。 |
| **LMCacheConnectorV1Impl** | vLLM統合の実装本体（`vllm_v1_adapter.py`）。LoadSpec/SaveSpecでロード・保存を管理。 |
| **LoadSpec** | ロード仕様。vLLMキャッシュ済みトークン数、LMCacheキャッシュ済みトークン数、ロード可否。 |
| **SaveSpec** | 保存仕様。`skip_leading_tokens`（キャッシュ済みプレフィックス長）、`can_save`（保存可否）。 |
| **ConnectorMetadata** | Scheduler→Worker間で渡されるメタデータ。各リクエストのtoken_ids, slot_mapping, LoadSpec, SaveSpecを含む。 |
| **kv_role** | `"kv_both"`（default）/`"kv_producer"`/`"kv_consumer"`。producer時はskip_leading_tokens=0。 |
| **LookupClient** | Scheduler側でキャッシュ存在確認を行うZMQベースクライアント。`lmcache_lookup_client.py` |
| **LookupServer** | Worker側でLookupClientからのZMQ REQ/REPを受け付け、StorageManager.async_lookup_and_prefetchを実行。 |
| **EventManager** | 非同期イベント（LOADING等）のFutureを管理。lookup_idでprefetch結果とretrieve消費を紐付け。 |
| **token_mask** | retrieve時のマスク。False=vLLMがキャッシュ済み（chunk_sizeの倍数に切り下げ）、True=LMCacheからロード対象。 |
| **ret_mask** | retrieve結果のマスク。True=LMCacheから実際に取得成功、False=未取得。Engine内部で構築。 |
| **write-back** | StorageManager.batched_get()がリモートバックエンドから取得した場合、自動的にLocalCPUBackendにコピーする動作。 |
| **get_block_mapping** | チャンクの所在バックエンドをprefix match方式で特定するStorageManagerメソッド。 |

## CacheBlend

| 用語 | 説明 |
|------|------|
| **CacheBlend** | 非プレフィックス部分のKVキャッシュも再利用する技術。重要トークンを再計算して品質保持。 |
| **Blender** | CacheBlendのblending計算を実行するコンポーネント。`lmcache/v1/compute/blend/` |
| **blend_recompute_ratios** | 再計算するトークンの割合。 |
| **blend_special_str** | セグメント分割用セパレータ文字列。デフォルト`" # # "`。 |

## 分散・マルチプロセス

| 用語 | 説明 |
|------|------|
| **CacheController** | 複数LMCacheインスタンス間のキャッシュ状態を中央管理するコントローラ。 |
| **LMCacheWorker** | CacheControllerと通信するワーカー。Heartbeat/Register/P2P Lookup。 |
| **MultiProcess Server** | ZMQ IPCベースの別プロセスLMCacheサーバー。SessionManager, GPUCacheContext管理。 |
| **BlendServer** | CacheBlend用MPサーバー。MPCacheEngine継承。 |
| **OffloadServer** | KVキャッシュオフロード用ZMQサーバー。 |
| **Disaggregated Prefill (PD)** | Prefill/Decode分離アーキテクチャ。NIXL経由でPD間転送。 |

## 設定キー（主要）

| 設定名 | デフォルト | 説明 |
|--------|-----------|------|
| `chunk_size` | 256 | チャンクのトークン数 |
| `local_cpu` | true | CPU バックエンド有効化 |
| `max_local_cpu_size` | 5.0 (GB) | CPUストレージ上限 |
| `local_disk` | None | ディスクパス（Noneで無効） |
| `remote_url` | None | リモートストレージURL |
| `remote_serde` | "naive" | リモート用Serde |
| `use_layerwise` | false | レイヤー単位転送 |
| `enable_blending` | false | CacheBlend有効化 |
| `enable_p2p` | false | P2P転送有効化 |
| `enable_pd` | false | Disaggregated Prefill |
| `enable_controller` | false | CacheController有効化 |
| `save_decode_cache` | false | Decodeフェーズのキャッシュも保存 |
