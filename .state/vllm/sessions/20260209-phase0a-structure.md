# セッション記録: vLLM全体構造の把握

> **日付**: 2026-02-09
> **フェーズ**: Phase 0a
> **命名規約**: `20260209-phase0a-structure.md`

## 目的

vLLMの全体像を把握し、以後の調査を効率化するための読解ガイド・概要ドキュメント・用語集を整備する。

## Phase完了条件の進捗

- [x] overview.mdが作成されている
- [x] 主要エントリポイントが3つ以上特定されている（CLI, LLM, AsyncLLM, OpenAI API）
- [x] glossary.mdに10用語以上登録されている（19用語）
- [x] reading-guide.mdに構造ルールが3つ以上記載されている（6ルール）

## 調査経路

1. README.md、pyproject.toml、ディレクトリ構成から全体像を把握
2. `vllm/engine/llm_engine.py` → v1への薄いラッパーであることを発見（最重要発見）
3. `vllm/v1/` の構造を詳細調査: engine → core → executor → worker 階層
4. `vllm/model_executor/models/` で241モデルファイルを確認
5. `vllm/distributed/kv_transfer/` でKV Transfer/LMCache統合の構造を確認
6. `vllm/v1/worker/gpu/mm/` でマルチモーダルキャッシュ関連を確認

## 発見

- **v1が本体**: `vllm/engine/` は `vllm/v1/engine/` への1行エイリアスラッパー [VERIFIED]
- **5層アーキテクチャ**: エントリポイント → エンジン → コア → 実行 → モデル
- **ZMQ IPC**: EngineCoreが別プロセスで動作し、ZMQで通信
- **KV Transfer**: `vllm/distributed/kv_transfer/` に複数バックエンド（LMCache, NIXL, P2P NCCL, Mooncake, Moriio）
- **モデル数**: 241ファイル。Gemma3をマルチモーダル対応リファレンス実装として選定
- **設定分割**: 旧`config.py`ではなく`config/`ディレクトリに分割。`VllmConfig`が集約
- **プラグインシステム**: `load_general_plugins()` がEngineCore初期化時に呼ばれる

## つまずき・未解決

- ZMQ IPC採用の設計判断の背景が不明
- v0→v1移行の経緯・時期が不明
- mm_cacheの詳細な動作メカニズムは未調査

## 成果物

- `docs/src/architecture/overview.md` — アーキテクチャ概要 [SHALLOW/VERIFIED]
- `docs/src/glossary.md` — 19用語の用語集 [SHALLOW/INFERRED]
- `.state/reading-guide.md` — 構造ルール6つ + ユーザー優先度
- `.claude/CLAUDE.md` — プロジェクト情報記入

## Phase完了判定

- このPhaseの完了条件を全て満たしたか: **Yes**（Phase 0a完了）
- Phase 0bは次セッションで実施（ユーザー優先度は本セッション中に確認済みのため、reading-guideに反映済み）

## 次回への引き継ぎ

- Phase 1（垂直スライス）を開始する
- 垂直スライスの候補をユーザーに提示して選択してもらう
- 候補1: テキスト推論のフルパス（リクエスト→Prefill→Decode→応答）
- 候補2: マルチモーダル推論のフルパス（画像付きリクエスト→エンコーダ→推論→応答）
