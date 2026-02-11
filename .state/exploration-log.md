# 探索ログ

## 現在のフェーズ

Phase 2: コンポーネント別深堀り（KVCacheManager DEEP 完了）

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
| Executor/Worker | [SHALLOW] | 2026-02-11 | `docs/src/components/executor/summary.md` |
| GPUModelRunner | [SHALLOW] | 2026-02-11 | `docs/src/components/gpu-model-runner/summary.md` |
| OutputProcessor | [SHALLOW] | 2026-02-11 | `docs/src/components/output-processor/summary.md` |
| モデル層 | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| KV Transfer/LMCache | [SHALLOW] | 2026-02-09 | `docs/src/glossary.md` |
| マルチモーダル | [SHALLOW] | 2026-02-09 | `docs/src/glossary.md` |

## セッション履歴

| 日付 | Phase | 概要 | 詳細 |
|------|-------|------|------|
| 2026-02-09 | 0a | 構造把握: 全体アーキテクチャ、主要コンポーネント特定、reading-guide作成 | `.state/sessions/20260209-phase0a-structure.md` |
| 2026-02-09 | 1a | 上流パス追跡: AsyncLLM→InputProcessor→EngineCoreClient→EngineCore到達 | `.state/sessions/20260209-phase1a-upstream-path.md` |
| 2026-02-11 | 1b | コアループ追跡: EngineCore.step()→Scheduler→KVCacheManager | `.state/sessions/20260211-phase1b-core-loop.md` |
| 2026-02-11 | 1c | 下流パス追跡: Executor→Worker→GPUModelRunner→OutputProcessor | `.state/sessions/20260211-phase1c-downstream-path.md` |
| 2026-02-11 | 2a | KVCacheManager深堀り: BlockPool、プレフィックスキャッシュ、アテンションタイプ別Manager（7種）。[MEDIUM]→[DEEP] | `.state/sessions/20260211-phase2a-kvcache-deep.md` |
