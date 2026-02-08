# 未解決の疑問

調査を駆動する疑問をここに蓄積する。解決したらチェックを入れ、回答のポインタを記載する。

## 優先

- [ ] ZMQ IPCを採用した理由は何か？ — EngineCoreが別プロセスで動作する設計判断の背景
- [ ] v0→v1の移行はいつ、なぜ行われたか？ — `vllm/engine/` がラッパーになった経緯
- [ ] KVCacheManagerとBlockPoolの関係は？ — ブロック割当アルゴリズムの詳細
- [ ] KV Transferの各バックエンド（LMCache, NIXL, P2P NCCL, Mooncake）の違い・使い分けは？
- [ ] mm_cache（マルチモーダルキャッシュ）はKVCacheManagerとどう連携するか？
- [ ] プラグインシステムの拡張ポイントはどこにあるか？ — `load_general_plugins()` の仕組み

## いつか調べる

- [ ] GPUModelRunnerが6277行もある理由 — 何がこのクラスに集約されているのか
- [ ] Schedulerのバッチサイズ決定ロジック — 何に基づいてバッチを構成するか
- [ ] FlashAttention vs FlashInfer の使い分け基準
- [ ] torch.compile統合（`vllm/compilation/`）の仕組み
