# 未解決の疑問

調査を駆動する疑問をここに蓄積する。解決したらチェックを入れ、回答のポインタを記載する。

## 優先

- [ ] KV Transferの各バックエンド（LMCache, NIXL, P2P NCCL, Mooncake）の違い・使い分けは？
- [ ] mm_cache（マルチモーダルキャッシュ）はKVCacheManagerとどう連携するか？
- [ ] プラグインシステムの拡張ポイントはどこにあるか？ — `load_general_plugins()` の仕組み
- [ ] GPUModelRunnerが6277行もある理由 — 何がこのクラスに集約されているのか
- [ ] batch_queue パイプライン並列化は実際にどう動作するか？（max_concurrent_batches > 1 時のオーバーラップ）
- [ ] block_size の設定方法とパフォーマンスへの影響は？
- [ ] async_scheduling と Speculative Decoding のドラフトトークンタイミングの相互作用は？

## 解決済み

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
- [ ] プリエンプション発生のメモリ圧力閾値の具体的な決定方法
