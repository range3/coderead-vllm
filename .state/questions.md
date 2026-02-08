# 未解決の疑問

調査を駆動する疑問をここに蓄積する。解決したらチェックを入れ、回答のポインタを記載する。

## 優先

- [ ] KVCacheManagerとBlockPoolの関係は？ — ブロック割当アルゴリズムの詳細
- [ ] KV Transferの各バックエンド（LMCache, NIXL, P2P NCCL, Mooncake）の違い・使い分けは？
- [ ] mm_cache（マルチモーダルキャッシュ）はKVCacheManagerとどう連携するか？
- [ ] プラグインシステムの拡張ポイントはどこにあるか？ — `load_general_plugins()` の仕組み
- [ ] Scheduler.schedule() のトークン予算割当はどのように動作するか？ — running/waitingの優先度
- [ ] SchedulerOutput に含まれる情報の全体像は？ — NewRequestData vs CachedRequestData の違い

## 部分的に解決

- [x] ZMQ IPCを採用した理由は何か？ — **回答**: EngineCoreが別プロセスで動作し、GIL回避とスケジューリング/GPU実行の並行を実現。ZMQ ROUTER/PULLソケット + msgpackシリアライゼーション。詳細は `docs/src/components/engine-core-client/summary.md`
- [x] v0→v1の移行はいつ、なぜ行われたか？ — **回答**: `vllm/engine/` がv1への1行エイリアス。v1が現行本体。移行の時期・理由は未調査だが、プロセス分離やContinuous Batching改善が動機と推測 [INFERRED]

## いつか調べる

- [ ] GPUModelRunnerが6277行もある理由 — 何がこのクラスに集約されているのか
- [ ] Schedulerのバッチサイズ決定ロジック — 何に基づいてバッチを構成するか
- [ ] FlashAttention vs FlashInfer の使い分け基準
- [ ] torch.compile統合（`vllm/compilation/`）の仕組み
- [ ] InputPreprocessor内部のトークナイザ呼び出しフロー詳細
- [ ] n>1サンプリング時のParentRequest/子リクエスト管理の仕組み
