# 未解決の疑問

調査を駆動する疑問をここに蓄積する。解決したらチェックを入れ、回答のポインタを記載する。

## 優先

- [ ] KV Transferの各バックエンド（LMCache, NIXL, P2P NCCL, Mooncake）の違い・使い分けは？
- [ ] mm_cache（マルチモーダルキャッシュ）はKVCacheManagerとどう連携するか？
- [ ] プラグインシステムの拡張ポイントはどこにあるか？ — `load_general_plugins()` の仕組み
- [ ] GPUModelRunnerの_build_attention_metadata()はKVCacheManagerのブロック情報をどう参照するか？（block_idsがSchedulerOutputに含まれ、GPUModelRunnerに渡される。詳細はGPUModelRunner深堀りで調査）
- [ ] FastIncrementalDetokenizer vs SlowIncrementalDetokenizer の実際のパフォーマンス差は？
- [ ] batch_queue パイプライン並列化は実際にどう動作するか？（max_concurrent_batches > 1 時のオーバーラップ）
- [x] block_size の設定方法とパフォーマンスへの影響は？ — **回答**: block_sizeはKVCacheSpecから取得され、モデルのアテンションタイプに依存。DCP/PCP > 1の場合は並列度倍に拡大。Hybrid modelでは異なるblock_sizeのグループが共存し、hash_block_size（最小のblock_size）でハッシュを計算後、BlockHashListWithBlockSizeで粒度変換。詳細は `docs/src/components/kv-cache-manager/prefix-cache.md` と `attention-type-managers.md`
- [ ] async_scheduling と Speculative Decoding のドラフトトークンタイミングの相互作用は？

## 解決済み

- [x] GPUModelRunnerが6277行もある理由 — **回答**: バッチ状態管理、入力準備、Attentionメタデータ、モデルフォワード（CUDAGraph対応）、サンプリング、KV Transfer、Speculative Decoding、PP、LoRA、マルチモーダルの10+責務を集約。詳細は `docs/src/components/gpu-model-runner/summary.md`
- [x] ZMQ IPCを採用した理由は何か？ — **回答**: EngineCoreが別プロセスで動作し、GIL回避とスケジューリング/GPU実行の並行を実現。ZMQ ROUTER/PULLソケット + msgpackシリアライゼーション。詳細は `docs/src/components/engine-core-client/summary.md`
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
