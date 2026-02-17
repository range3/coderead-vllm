# 未解決の疑問

調査を駆動する疑問をここに蓄積する。解決したらチェックを入れ、回答のポインタを記載する。

## 優先

- [x] KV Transferの各バックエンド（LMCache, NIXL, P2P NCCL, Mooncake）の違い・使い分けは？ — **回答**: KVConnectorFactoryに10個登録。LMCache=チャンク単位保存・3層ストレージ（CPU/Disk/Remote）・15+リモートコネクタ、NIXL=RDMA高速GPU間転送・KVキャッシュ事前登録、P2P NCCL=NCCL直接GPU転送、Mooncake=分散フレームワーク統合、Offloading=CPU/ディスクオフロード、Multi=複数バックエンド束ね。詳細は `docs/src/components/kv-transfer/summary.md` と `docs/src/investigations/lmcache-integration.md`
- [x] mm_cache（マルチモーダルキャッシュ）はKVCacheManagerとどう連携するか？ — **回答**: MMキャッシュはKVCacheManagerとは独立。ProcessorCache（P0、HF処理結果）とEncoderCacheManager（P1、エンコーダ出力の論理管理）+ encoder_cache（GPU、テンソル）の3層構造。KVCacheManagerはデコーダ側のKVキャッシュのみ管理。ただしプレフィックスキャッシュのExtra Keysとしてmm_hashが使われ、同じ画像のリクエストはKVキャッシュのプレフィックスも共有可能。詳細は `docs/src/components/multimodal/summary.md`
- [ ] プラグインシステムの拡張ポイントはどこにあるか？ — `load_general_plugins()` の仕組み
- [x] GPUModelRunnerの_build_attention_metadata()はKVCacheManagerのブロック情報をどう参照するか？ — **回答**: 4段変換パス。`_update_states()`でSchedulerOutput.new_block_idsをCachedRequestState.block_ids+InputBatch.block_tableに取込→`BlockTable.compute_slot_mapping()`で`block_number*block_size+offset`の変換→`commit_block_table()`/`commit_slot_mapping()`でCpuGpuBuffer DMA転送→`_build_attention_metadata()`でCommonAttentionMetadata構築→per-layer AttentionMetadata。詳細は `docs/src/components/gpu-model-runner/kv-cache-interface.md`
- [ ] ForwardContextでのslot_mappings_by_layer消費: reshape_and_cache()の具体的な実装場所は？
- [ ] CUDAGraph FULL mode時にslot_mappingsは毎step書き直されるか、それとも固定か？
- [ ] FastIncrementalDetokenizer vs SlowIncrementalDetokenizer の実際のパフォーマンス差は？
- [ ] batch_queue パイプライン並列化は実際にどう動作するか？（max_concurrent_batches > 1 時のオーバーラップ）
- [x] block_size の設定方法とパフォーマンスへの影響は？ — **回答**: block_sizeはKVCacheSpecから取得され、モデルのアテンションタイプに依存。DCP/PCP > 1の場合は並列度倍に拡大。Hybrid modelでは異なるblock_sizeのグループが共存し、hash_block_size（最小のblock_size）でハッシュを計算後、BlockHashListWithBlockSizeで粒度変換。詳細は `docs/src/components/kv-cache-manager/prefix-cache.md` と `attention-type-managers.md`
- [ ] async_scheduling と Speculative Decoding のドラフトトークンタイミングの相互作用は？

## 解決済み

- [x] GPUModelRunnerが6277行もある理由 — **回答**: バッチ状態管理、入力準備、Attentionメタデータ、モデルフォワード（CUDAGraph対応）、サンプリング、KV Transfer、Speculative Decoding、PP、LoRA、マルチモーダルの10+責務を集約。詳細は `docs/src/components/gpu-model-runner/summary.md`
- [x] ZMQ IPCを採用した理由は何か？ — **回答**: EngineCoreが別プロセスで動作し、GIL回避とスケジューリング/GPU実行の並行を実現。ZMQ ROUTER/PULLソケット + msgpackシリアライゼーション。EngineCore↔WorkerはSharedMemory MQ（低レイテンシ）、Worker間はNCCL（GPU直接通信）。通信方式選択理由の詳細は `docs/src/investigations/process-architecture.md`。ZMQ通信パターンの体系的調査は `docs/src/investigations/zmq-communication-patterns.md`
- [x] v0→v1の移行はいつ、なぜ行われたか？ — **回答**: `vllm/engine/` がv1への1行エイリアス。v1が現行本体。移行の時期・理由は未調査だが、プロセス分離やContinuous Batching改善が動機と推測 [INFERRED]
- [x] Scheduler.schedule() のトークン予算割当はどのように動作するか？ — **回答**: `token_budget = max_num_scheduled_tokens` で初期化し、Phase 1（RUNNING）→ Phase 2（WAITING）で各リクエストのスケジュール時に消費。Unified Compute Modelで統一管理。詳細は `docs/src/components/scheduler/summary.md`
- [x] SchedulerOutput に含まれる情報の全体像は？ — **回答**: 15フィールド。NewRequestData（初回フルデータ）と CachedRequestData（差分のみ）の2種。詳細は `docs/src/architecture/data-flow.md` の境界データ構造セクション
- [x] KVCacheManagerとBlockPoolの関係は？ — **回答**: KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool の4層階層。BlockPoolが物理ブロック管理（割当・解放・LRU Eviction）を担当。参照カウント方式。詳細は `docs/src/components/kv-cache-manager/summary.md`
- [x] Schedulerのバッチサイズ決定ロジック — **回答**: `max_num_scheduled_tokens`（トークン予算）と `max_num_seqs`（リクエスト数上限）で制約。予算消費は単調減少。詳細は `docs/src/components/scheduler/summary.md`

## いつか調べる

- [ ] FlashAttention vs FlashInfer の使い分け基準
- [ ] torch.compile統合（`vllm/compilation/`）の仕組み
- [ ] InputPreprocessor内部のトークナイザ呼び出しフロー詳細
- [ ] n>1サンプリング時のParentRequest/子リクエスト管理の仕組み
- [x] プリエンプション発生のメモリ圧力閾値の具体的な決定方法 — **回答**: 明示的な閾値はない。allocate_slots()で `num_blocks_to_allocate > block_pool.get_num_free_blocks()` の場合にNoneを返し、Schedulerがプリエンプション（RUNNING）またはスキップ（WAITING）を実行。空きブロック数は動的に変化し、Evictionも含めた現在の空き状況で判定。詳細は `docs/src/components/kv-cache-manager/summary.md`
- [ ] HybridKVCacheCoordinatorの反復固定点アルゴリズムは実際のモデルで何回イテレーションするか？
- [ ] BlockHashToBlockMapのUnion型最適化の実測パフォーマンス差は？
- [ ] ProcessorCacheのshm（共有メモリ）モードのSingleWriterShmRingBufferの具体的な動作は？
- [x] マルチモーダルのプレフィックスキャッシュとProcessorCacheの相互作用の詳細は？（同じ画像のリクエストでKVキャッシュヒットする条件） — **回答**: 3つのキャッシュは独立に動作し、共通のmm_hashをキー基盤として共有。同一画像・同一プロンプト→全ヒット、同一画像・異なるプロンプト→ProcessorCache+EncoderCacheヒット+KVプレフィックスは画像ブロック部分のみヒット可能。extra_keysにidentifierが含まれるため異なる画像は必ずミス。詳細は `docs/src/investigations/gemma3-vision-caches.md`
- [ ] 複数画像入力時のエンコーダバッチ処理のパフォーマンス特性は？
- [ ] Mooncake ECConnector統一案はSHMConnectorを置き換えるか？（#33714 議論中）
- [ ] エンコーダキャッシュのdict→事前割り当て型移行はECConnectorBaseのインタフェースに影響するか？
- [ ] ECキャッシュ解放メカニズムはどのように実装される予定か？（#32659）
- [ ] NixlConnectorのRDMA事前登録（register_kv_caches）はどのような最適化をもたらすか？
- [ ] cross-layer blocks（prefer_cross_layer_blocks=True）を使うコネクタはどれか、実際の性能差は？
- [ ] LMCacheのLookupClient/Serverの分散キャッシュ問い合わせの具体的な通信プロトコルは？
- [x] LMCache CacheBlend（enable_blending）の動作メカニズムは？（partial KV reuse） — **回答**: CacheBlendはRAGチャンク間のKV再利用を実現。セパレータで段落分割→段落単位KV保存・ルックアップ→topk重要token同定（K差分L2ノルム）→選択的KV再計算。vLLMモデルオブジェクトに直接アクセスする独自forward pathを持ち、vLLM本体へのad-hocパッチが必要（プラグインのみでは不完結）。対応モデルはLlama/Qwen2/Qwen3の3種のみ。詳細は `docs/src/investigations/cacheblend-implementation.md`
- [ ] KV Transfer使用時のSchedulerのWAITING_FOR_REMOTE_KVS→WAITING遷移のレイテンシ影響は？
