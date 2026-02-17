# 探索ログ

## 現在のフェーズ

Phase 2/3 並行: コンポーネント別深堀り + 横断調査（KVCacheManager DEEP完了、マルチモーダル MEDIUM完了、EncoderCache MEDIUM完了、ECConnector MEDIUM完了、Executor MEDIUM完了、GPUModelRunner MEDIUM完了、KV Transfer MEDIUM完了、ZMQ通信パターン横断調査完了）

## カバレッジマップ

| 領域 | 深度 | 最終更新 | 関連ドキュメント |
|------|------|---------|----------------|
| 全体アーキテクチャ | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| テキスト推論データフロー | [MEDIUM] | 2026-02-11 | `docs/src/architecture/data-flow.md` |
| エントリポイント (AsyncLLM/LLM) | [SHALLOW] | 2026-02-09 | `docs/src/components/entrypoint/summary.md` |
| InputProcessor | [SHALLOW] | 2026-02-09 | `docs/src/components/input-processor/summary.md` |
| EngineCoreClient (ZMQ IPC) | [SHALLOW] | 2026-02-09 | `docs/src/components/engine-core-client/summary.md` |
| EngineCore | [MEDIUM] | 2026-02-11 | `docs/src/components/engine-core/summary.md` |
| Scheduler | [MEDIUM] | 2026-02-11 | `docs/src/components/scheduler/summary.md` |
| KVCacheManager | [DEEP] | 2026-02-11 | `docs/src/components/kv-cache-manager/summary.md` + 3 サブドキュメント |
| Executor/Worker | [MEDIUM] | 2026-02-14 | `docs/src/components/executor/summary.md` |
| プロセスアーキテクチャ（TP=2） | [DEEP] | 2026-02-14 | `docs/src/investigations/process-architecture.md` |
| GPUModelRunner | [MEDIUM] | 2026-02-15 | `docs/src/components/gpu-model-runner/summary.md` + 2 サブドキュメント |
| OutputProcessor | [SHALLOW] | 2026-02-11 | `docs/src/components/output-processor/summary.md` |
| モデル層 | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| KV Transfer/LMCache | [MEDIUM] | 2026-02-15 | `docs/src/components/kv-transfer/summary.md` + `docs/src/investigations/lmcache-integration.md` + `docs/src/investigations/cacheblend-implementation.md` |
| EncoderCache | [MEDIUM] | 2026-02-14 | `docs/src/components/encoder-cache/summary.md` |
| ECConnector (Encoder Cache Transfer) | [MEDIUM] | 2026-02-14 | `docs/src/components/ec-connector/summary.md` + investigations 2件 |
| マルチモーダル | [MEDIUM→DEEP(§3)] | 2026-02-17 | `docs/src/components/multimodal/summary.md` + 3 サブドキュメント |
| ZMQ通信パターン（横断調査） | [MEDIUM] | 2026-02-18 | `docs/src/investigations/zmq-communication-patterns.md` |

## セッション履歴

| 日付 | Phase | 概要 | 詳細 |
|------|-------|------|------|
| 2026-02-09 | 0a | 構造把握: 全体アーキテクチャ、主要コンポーネント特定、reading-guide作成 | `.state/sessions/20260209-phase0a-structure.md` |
| 2026-02-09 | 1a | 上流パス追跡: AsyncLLM→InputProcessor→EngineCoreClient→EngineCore到達 | `.state/sessions/20260209-phase1a-upstream-path.md` |
| 2026-02-11 | 1b | コアループ追跡: EngineCore.step()→Scheduler→KVCacheManager | `.state/sessions/20260211-phase1b-core-loop.md` |
| 2026-02-11 | 1c | 下流パス追跡: Executor→Worker→GPUModelRunner→OutputProcessor | `.state/sessions/20260211-phase1c-downstream-path.md` |
| 2026-02-11 | 2a | KVCacheManager深堀り: BlockPool、プレフィックスキャッシュ、アテンションタイプ別Manager（7種）。[MEDIUM]→[DEEP] | `.state/sessions/20260211-phase2a-kvcache-deep.md` |
| 2026-02-11 | 2b | マルチモーダル画像推論パス: ProcessorCache(4種)、MMHasher、EncoderCacheManager、GPUModelRunnerエンコーダ実行、Gemma3(SiglipVisionModel+Projector)、masked_scatter_マージ。[SHALLOW]→[MEDIUM] | `.state/sessions/20260211-phase2b-multimodal.md` |
| 2026-02-11 | 2b+ | Gemma3ビジョンパイプライン形状フロー調査（外部調査）。HFモデル設定・transformersコード配置。config.json導出値、Pan-and-Scan 2ケース比較 | `docs/src/investigations/gemma3-vision-pipeline.md` |
| 2026-02-11 | 2b++ | Gemma3ビジョンパイプラインのキャッシュ機構調査。ProcessorCache(CPU/blake3)、EncoderCache(GPU/identifier)、KVプレフィックスキャッシュ(GPU/extra_keys)の3層。各キャッシュのハッシュ入力・保存値・スキップ処理を特定 | `docs/src/investigations/gemma3-vision-caches.md` |
| 2026-02-14 | 2c | EncoderCache永続化・階層キャッシュ化の実現可能性調査。ECConnector既存インフラの発見（KV Transferとは独立した専用枠組み）。FIFO→LRU変更は1ファイル。ECExampleConnector参照実装分析。カスタムECConnector実装ガイド | `docs/src/investigations/encoder-cache-persistence.md` |
| 2026-02-14 | 2c+ | ECConnector GitHub議論調査。EPD分離基盤→Encoder-only→ec_both→SHMConnector/Mooncake統一案の開発経緯。未解決課題（キャッシュ解放、事前割り当て、MM前処理重複排除）。主要コントリビューター特定 | `docs/src/investigations/ec-connector-github-discussions.md` |
| 2026-02-14 | 2d | EncoderCache・ECConnectorコンポーネント文書化。submodule最新化後のコード再調査。EncoderCacheManager（FIFO遅延解放、共有キャッシュ、EncoderDecoderCacheManager）、ECConnector（2ロール分離、Mixin統合、Producer専用モード、未実装機能5点特定） | `.state/sessions/20260214-phase2d-encoder-cache-ec-connector.md` |
| 2026-02-14 | 2e | プロセスアーキテクチャ調査（TP=2構成）。4プロセス構成、3種通信（ZMQ/SharedMemory MQ/NCCL）、ShmRingBufferロックフリー設計、通信方式選択理由。Executor [SHALLOW]→[MEDIUM] 昇格 | `.state/sessions/20260214-phase2e-process-architecture.md` |
| 2026-02-14 | 2e+ | SharedMemory MQ深堀り + Worker→EngineCore結果返却パス。MessageQueue内部（pickle5 oob、バイトフォーマット、メモリフェンスプロトコル、SpinTimer）、response_mq構成、output_rankフィルタリング、async_scheduling、non_block/FutureWrapper。process-architecture.md [MEDIUM]→[DEEP] | `.state/sessions/20260214-phase2e+-shm-mq-deep.md` |
| 2026-02-15 | 2f | GPUModelRunner深堀り。KVCache-GPU Interface（ブロックID取込→BlockTable→slot_mapping→DMA→AttentionMetadata 4段変換）、InputBatch永続バッチ（CachedRequestState/InputBatch/MultiGroupBlockTable/CpuGpuBuffer/condense）、CUDAGraph統合（3モード/CudagraphDispatcher/パディング）。summary.md [SHALLOW]→[MEDIUM] 昇格 | `.state/sessions/20260215-phase2f-gpu-model-runner.md` |
| 2026-02-15 | 2g | KV Transfer / LMCache調査。KVConnectorBase_V1（7 abstract、2ロール分離）、KVConnectorFactory（10コネクタ）、Scheduler統合（WAITING_FOR_REMOTE_KVS）、Worker/Mixin統合、KV Cache Events、LMCacheチャンク単位保存・3層ストレージ・vLLMアダプタ。[SHALLOW]→[MEDIUM] 昇格 | `.state/sessions/20260215-phase2g-kv-transfer-lmcache.md` |
| 2026-02-15 | 2g+ | CacheBlend実装調査。独自forward path（LMCBaseModel.compute_layer）、重要token同定（K差分L2ノルムtopk）、VLLMBufferLayerwiseGPUConnector（RoPE補正+パイプライン）、vLLM本体パッチ必須（VLLMModelTracker登録）、対応モデル3種のみ、BlendServer段落分割、制約多数 | `.state/sessions/20260215-phase2g+-cacheblend.md` |
| 2026-02-17 | 2h | mm_hash計算方法調査。hash_kwargs()/serialize_item()/iter_item_to_bytes()の3層構造、画像3シリアライズパス（EXIF UUID/MediaWithBytes/ピクセル）、_hash_mm_items()のmm_uuids分岐、identifier vs mm_hashの使い分け、プレフィックスキャッシュextra_keys連携、Gemma3はデフォルト実装。mm-processing.md §3を[MEDIUM]→[DEEP]昇格 | `.state/sessions/20260217-phase2h-mm-hash.md` |
| 2026-02-18 | 3a | ZMQ通信パターン横断調査。16ファイル5カテゴリ（Frontend↔EngineCore/DPCoordinator/ShmRingBufferフォールバック/KV Events/KV Transfer）、10種ソケットタイプ使用一覧、信頼性分析（HWM=0+IPC+プロセス監視でコア通信は実質喪失なし、補助パスはベストエフォート+リカバリ） | `.state/sessions/20260218-phase3a-zmq-patterns.md` |
