# コンテキスト索引

各ドキュメントの内容を1行で要約。セッション開始時に読み、必要なドキュメントだけを選択的に読み込む。

## アーキテクチャ

| ドキュメント                         | 内容       | 深度 | 確信度 | 最終更新 |
| ------------------------------------ | ---------- | ---- | ------ | -------- |
| `docs/src/architecture/overview.md`  | vLLM全体のアーキテクチャ概要。5層構造（エントリポイント→エンジン→コア→実行→モデル）、v1が本体、ZMQ IPC分離 | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/architecture/data-flow.md` | テキスト推論データフロー + マルチモーダル差分セクション。境界データ構造5つ、上流パス、コアループ、下流パス、MM推論との差分 | [MEDIUM] | [VERIFIED] | 2026-02-11 |

## コンポーネント

| ドキュメント | 内容 | 深度 | 確信度 | 最終更新 |
| ------------ | ---- | ---- | ------ | -------- |
| `docs/src/components/entrypoint/summary.md` | AsyncLLM/LLMエントリポイント。generate()→add_request()→InputProcessor→EngineCoreClient。同期/非同期パスの違い | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/components/input-processor/summary.md` | InputProcessor。prompt→トークナイズ→SamplingParams正規化→EngineCoreRequest構築 | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/components/engine-core-client/summary.md` | EngineCoreClient。ZMQ ROUTER/PULL + msgpackによるプロセス間通信。クライアント階層（Async/Sync/DP） | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/components/engine-core/summary.md` | EngineCore。step()サイクル（schedule→execute→update）、KVキャッシュ初期化フロー、batch_queueパイプライン並列化、async_scheduling | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/scheduler/summary.md` | Scheduler。3フェーズschedule()（RUNNING→WAITING→Output構築）、Unified Compute Model、トークン予算、プリエンプション、Requestステータス遷移、update_from_output() | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/kv-cache-manager/summary.md` | KVCacheManager。4層階層、Coordinator 3種選択ロジック、KVCacheBlocks dataclass、allocate_slots() 5段階フロー、KVキャッシュグループ概念、アテンションタイプ7種、3サブドキュメントへのリンク | [DEEP] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/kv-cache-manager/block-pool.md` | BlockPool物理ブロック管理。KVCacheBlock（6フィールド、ライフサイクル）、FreeKVCacheBlockQueue（双方向リンクリスト、O(1) remove）、BlockHashToBlockMap（Union型最適化、重複排除なし）、null_block、Eviction、KV Cache Events、メトリクス | [DEEP] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/kv-cache-manager/prefix-cache.md` | プレフィックスキャッシュ。ハッシュチェーン計算（sha256_cbor等4種）、NONE_HASH、BlockHash型階層、Extra Keys（MM/LoRA/salt/embeds）、リクエストブロックハッシャー（遅延・増分）、BlockHashListWithBlockSize、Lookupアルゴリズム4種、Hybrid fixed-point | [DEEP] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/kv-cache-manager/attention-type-managers.md` | アテンションタイプ別Manager 7種。基底クラス（req_to_blocks/num_cached_block）、FullAttention（左→右scan）、SlidingWindow（右→左contiguous）、ChunkedLocal（chunk境界）、Mamba（align/none、状態管理）、CrossAttention（キャッシュなし）、SinkFullAttention（sink事前確保）、spec_manager_map | [DEEP] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/executor/summary.md` | Executor。collective_rpc()委譲パターン、UniProc/Multiproc/Ray 3実装、MultiprocExecutor MessageQueue詳細（ShmRingBuffer、ロックフリー、24MiBチャンク）、Worker起動・ビジーループ、Pipeline Parallelism対応 | [MEDIUM] | [VERIFIED] | 2026-02-14 |
| `docs/src/components/gpu-model-runner/summary.md` | GPUModelRunner。2フェーズ実行パターン、KVCache-GPU Interface（4段変換）、CUDAGraph統合（3モード）、InputBatch永続バッチ、6300行セクション内訳。2サブドキュメントへのリンク | [MEDIUM] | [VERIFIED] | 2026-02-15 |
| `docs/src/components/gpu-model-runner/kv-cache-interface.md` | KVCache-GPU Interface。ブロックID取込（3ケース）→BlockTable→slot_mapping計算式→DMA転送→CommonAttentionMetadata→per-layer AttentionMetadata。Hybrid Block対応、CpuGpuBuffer、2形式スロットマッピング | [MEDIUM] | [VERIFIED] | 2026-02-15 |
| `docs/src/components/gpu-model-runner/input-batch.md` | InputBatch永続バッチ。CachedRequestState（論理状態）、InputBatch（物理バッチテンソル群）、add/remove/condense、MultiGroupBlockTable、永続バッチ最適化 | [MEDIUM] | [VERIFIED] | 2026-02-15 |
| `docs/src/components/output-processor/summary.md` | OutputProcessor。process_outputs()フロー、Detokenizer階層（Fast/Slow）、停止文字列判定、LogprobsProcessor、RequestOutputKind 3モード | [SHALLOW] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/encoder-cache/summary.md` | EncoderCache。2層構造（Scheduler側EncoderCacheManager論理管理+Worker側GPU物理ストレージ）、FIFO遅延解放Eviction、共有キャッシュ（mm_hash基盤）、EncoderDecoderCacheManager暫定実装、ECConnector連携 | [MEDIUM] | [VERIFIED] | 2026-02-14 |
| `docs/src/components/ec-connector/summary.md` | ECConnector。エンコーダキャッシュ外部転送プラグインフレームワーク。2ロール分離（Scheduler/Worker）、ECConnectorBase（5抽象メソッド）、ECTransferConfig（3ロール）、ECConnectorFactory（静的+動的登録）、ECExampleConnector参照実装、ECConnectorModelRunnerMixin、Producer専用モード、未実装機能5点 | [MEDIUM] | [VERIFIED] | 2026-02-14 |
| `docs/src/components/kv-transfer/summary.md` | KV Transfer。デコーダKVキャッシュ転送プラグインフレームワーク。KVConnectorBase_V1（7 abstract、2ロール分離）、KVConnectorFactory（10コネクタ）、Scheduler統合（WAITING_FOR_REMOTE_KVS、外部キャッシュ問い合わせ、遅延解放）、Worker/Mixin統合（レイヤー別パイプライニング）、KV Cache Events、cross-layer blocks、KVTransferConfig、ECConnector比較表 | [MEDIUM] | [VERIFIED] | 2026-02-15 |
| `docs/src/components/multimodal/summary.md` | マルチモーダル処理パイプライン全体像。3層キャッシュ構造、テキスト推論との差分、Gemma3固有特徴。3サブドキュメントへのリンク | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/multimodal/mm-processing.md` | フロントエンドMM処理。チャットテンプレート適用、プレースホルダー展開、トークン列構造、MMHasher(blake3)、ProcessorCache 4種(processor_only/lru/shm/none)、P0-P1キャッシュ整合性、MultiModalFeatureSpec構築 | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/multimodal/mm-engine-gpu.md` | バックエンドMM処理。EncoderCacheManager(RefCount+FIFO遅延Eviction)、Schedulerエンコーダ予算管理、GPUModelRunnerのencoder_cache/execute/gather/merge | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/multimodal/gemma3-vision.md` | Gemma3ビジョン。SiglipVisionModel(パッチ埋め込み→双方向Transformer)、Gemma3MultiModalProjector(AvgPool→RMSNorm→Linear)、Pan-and-Scan、masked_scatter_マージ | [MEDIUM] | [VERIFIED] | 2026-02-11 |

## 調査報告 (Investigations)

| ドキュメント | 内容 | 深度 | 確信度 | 最終更新 |
| ------------ | ---- | ---- | ------ | -------- |
| `docs/src/investigations/gemma3-vision-pipeline.md` | Gemma3 27Bビジョンパイプライン形状フロー。config.json導出値、Pan-and-Scan設定、APIリクエスト→デコーダ入力のフルステップ（ケース1: PaS無効=256トークン、ケース2: PaS有効=768トークン）、CPU/GPU処理フロー図 | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/investigations/gemma3-vision-caches.md` | Gemma3ビジョンパイプラインの3層キャッシュ機構。ProcessorCache(CPU, blake3, Step3スキップ)、EncoderCache(GPU, identifier, Step4+5+6スキップ)、KVプレフィックスキャッシュ(GPU, extra_keys, Step7+8部分スキップ)。キャッシュ間のキー共有(mm_hash基盤)、4シナリオ別動作例 | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/investigations/encoder-cache-persistence.md` | EncoderCache永続化・階層キャッシュ化の調査報告。ECConnector既存インフラの発見と分析（ECConnectorBase/ECExampleConnector/ECConnectorFactory/ECTransferConfig）。KV Transferとの比較（ECConnectorが正解）。FIFO→LRU変更設計（encoder_cache_manager.pyの2メソッド修正）。2層キャッシュ設計（L1:GPU/LRU + L2:ECConnector/Storage）。カスタムECConnector実装ガイド | [MEDIUM] | [VERIFIED] | 2026-02-14 |
| `docs/src/investigations/ec-connector-github-discussions.md` | ECConnector GitHub議論調査。EPD分離基盤(#25233)、Encoder-onlyモード(#30242)、ec_bothロール(#34182)のマージ済み設計。SHMConnector vs Mooncake統一案の進行中議論。エンコーダキャッシュ事前割り当て問題。MM前処理重複排除RFC。主要コントリビューター・タイムライン・未解決課題一覧 | [MEDIUM] | [VERIFIED] | 2026-02-14 |
| `docs/src/investigations/cacheblend-github-discussions.md` | CacheBlend GitHub議論調査。オンライン推論(vllm serve)未対応（8ヶ月間）、トークン化不一致が根本障壁。vLLM本体RFC#25950（サブリクエスト分割アプローチ、コード未公開）。LMCache側の品質バグ多数（ガーブル出力、保存漏れ、layerwise破損）。バージョン互換性マトリクス | [MEDIUM] | [VERIFIED] | 2026-02-14 |
| `docs/src/investigations/process-architecture.md` | プロセスアーキテクチャ（TP=2構成）。4プロセス構成、3種通信、ShmRingBufferロックフリー設計、MessageQueue詳細（enqueue/dequeueバイトフォーマット、pickle5 oob buffers、メモリフェンスプロトコル、SpinTimer）、Worker→EngineCore結果返却パス（response_mq構成、output_rankフィルタリング、async_scheduling、non_block/FutureWrapper） | [DEEP] | [VERIFIED] | 2026-02-14 |
| `docs/src/investigations/lmcache-integration.md` | LMCache統合調査。チャンク単位KV保存（256トークン/チャンク）、CacheEngineKey（プレフィックスハッシュ）、3層ストレージ階層（CPU/Disk/Remote）、15+リモートコネクタ。vLLMアダプタ（native/latest 2パス分岐）、RequestTracker/ReqMeta/LoadSpec/SaveSpec、GPUConnector 3種、KV形状、セーブ判定ロジック、Disaggregated Serving、設定体系 | [MEDIUM] | [VERIFIED] | 2026-02-15 |

## 外部リソース (target/ 内参照用)

| パス | 内容 | 用途 |
| ---- | ---- | ---- |
| `target/gemma3-27b-it/` | Gemma3-27b-it HF公開モデル設定ファイル群（config.json, preprocessor_config.json, chat_template.json等8ファイル。weightなし） | モデルパラメータ・トークンID・Pan-and-Scan設定の一次情報源 |
| `target/transformers/src/transformers/models/gemma3/` | HF transformers Gemma3実装（8ファイル: configuration, modeling, processing, image_processing等） | vLLMが直接呼び出す上流コード。ProcessorCacheがGemma3Processorを呼び、Gemma3ProcessorがGemma3ImageProcessorを呼ぶ。processing_gemma3.pyのデフォルト値定義・boi展開ロジック、image_processing_gemma3.pyのPan-and-Scan・リサイズ実装が特に重要 |
| `target/LMCache/lmcache/` | LMCacheライブラリ本体。v1/cache_engine.py（中核エンジン）、v1/storage_backend/（3層ストレージ+15+コネクタ）、integration/vllm/（latest版vLLMアダプタ） | KV Transferの主要バックエンド実装。チャンク単位KV保存の仕組み、ストレージバックエンド設計の参照 |

## 付録

| ドキュメント                      | 内容             | 深度 | 確信度 | 最終更新 |
| --------------------------------- | ---------------- | ---- | ------ | -------- |
| `docs/src/glossary.md`            | 46用語。Phase 2gで追加: KVConnectorBase_V1、KVConnectorFactory、KVTransferConfig、KVConnectorModelRunnerMixin、Disaggregated Prefill、WAITING_FOR_REMOTE_KVS、KV Cache Events、CacheEngineKey | [SHALLOW] | [VERIFIED] | 2026-02-15 |
| `.state/questions.md`             | 未解決の疑問     | -    | -      | 2026-02-11 |
| `docs/src/appendix/file-index.md` | 主要ファイル索引 | -    | -      | -        |
| `.state/reading-guide.md`         | 構造ルール6つ + ユーザー優先度。v1優先、gemma3リファレンス、CUDA中心 | [INFERRED] | [INFERRED] | 2026-02-09 |
