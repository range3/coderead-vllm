# 探索ログ

## 現在のフェーズ

Phase 1: 垂直スライス（セッション1/3 完了）

## カバレッジマップ

| 領域 | 深度 | 最終更新 | 関連ドキュメント |
|------|------|---------|----------------|
| 全体アーキテクチャ | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| テキスト推論データフロー | [SHALLOW] | 2026-02-09 | `docs/src/architecture/data-flow.md` |
| エントリポイント (AsyncLLM/LLM) | [SHALLOW] | 2026-02-09 | `docs/src/components/entrypoint/summary.md` |
| InputProcessor | [SHALLOW] | 2026-02-09 | `docs/src/components/input-processor/summary.md` |
| EngineCoreClient (ZMQ IPC) | [SHALLOW] | 2026-02-09 | `docs/src/components/engine-core-client/summary.md` |
| エンジン層 (v1) | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| コア層 (Scheduler/KVCache) | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| 実行層 (Executor/Worker) | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| モデル層 | [SHALLOW] | 2026-02-09 | `docs/src/architecture/overview.md` |
| KV Transfer/LMCache | [SHALLOW] | 2026-02-09 | `docs/src/glossary.md` |
| マルチモーダル | [SHALLOW] | 2026-02-09 | `docs/src/glossary.md` |

## セッション履歴

| 日付 | Phase | 概要 | 詳細 |
|------|-------|------|------|
| 2026-02-09 | 0a | 構造把握: 全体アーキテクチャ、主要コンポーネント特定、reading-guide作成 | `.state/sessions/20260209-phase0a-structure.md` |
| 2026-02-09 | 1a | 上流パス追跡: AsyncLLM→InputProcessor→EngineCoreClient→EngineCore到達 | `.state/sessions/20260209-phase1a-upstream-path.md` |
