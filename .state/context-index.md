# コンテキスト索引

各ドキュメントの内容を1行で要約。セッション開始時に読み、必要なドキュメントだけを選択的に読み込む。

## アーキテクチャ

| ドキュメント                         | 内容       | 深度 | 確信度 | 最終更新 |
| ------------------------------------ | ---------- | ---- | ------ | -------- |
| `docs/src/architecture/overview.md`  | vLLM全体のアーキテクチャ概要。5層構造（エントリポイント→エンジン→コア→実行→モデル）、v1が本体、ZMQ IPC分離 | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/architecture/data-flow.md` | テキスト推論データフロー完成版。境界データ構造5つ、上流パス、コアループ、下流パス（Executor委譲・GPUModelRunner 2フェーズ実行・OutputProcessorデトークナイズ）、Prefill vs Decode、コンポーネント優先度確定 | [MEDIUM] | [VERIFIED] | 2026-02-11 |

## コンポーネント

| ドキュメント | 内容 | 深度 | 確信度 | 最終更新 |
| ------------ | ---- | ---- | ------ | -------- |
| `docs/src/components/entrypoint/summary.md` | AsyncLLM/LLMエントリポイント。generate()→add_request()→InputProcessor→EngineCoreClient。同期/非同期パスの違い | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/components/input-processor/summary.md` | InputProcessor。prompt→トークナイズ→SamplingParams正規化→EngineCoreRequest構築 | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/components/engine-core-client/summary.md` | EngineCoreClient。ZMQ ROUTER/PULL + msgpackによるプロセス間通信。クライアント階層（Async/Sync/DP） | [SHALLOW] | [VERIFIED] | 2026-02-09 |
| `docs/src/components/engine-core/summary.md` | EngineCore。step()サイクル（schedule→execute→update）、KVキャッシュ初期化フロー、batch_queueパイプライン並列化、async_scheduling | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/scheduler/summary.md` | Scheduler。3フェーズschedule()（RUNNING→WAITING→Output構築）、Unified Compute Model、トークン予算、プリエンプション、Requestステータス遷移、update_from_output() | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/kv-cache-manager/summary.md` | KVCacheManager。階層設計（→Coordinator→SingleType→BlockPool）、allocate_slots()ブロック配置図、プレフィックスキャッシュ、参照カウント・Eviction、スライディングウィンドウ | [MEDIUM] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/executor/summary.md` | Executor。collective_rpc()委譲パターン、UniProc/Multiproc/Ray 3実装、Worker委譲フロー、Pipeline Parallelism対応 | [SHALLOW] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/gpu-model-runner/summary.md` | GPUModelRunner。2フェーズ実行パターン（execute_model→sample_tokens）、ExecuteModelState、6300行の内訳、Phase 2深堀り候補 | [SHALLOW] | [VERIFIED] | 2026-02-11 |
| `docs/src/components/output-processor/summary.md` | OutputProcessor。process_outputs()フロー、Detokenizer階層（Fast/Slow）、停止文字列判定、LogprobsProcessor、RequestOutputKind 3モード | [SHALLOW] | [VERIFIED] | 2026-02-11 |

## 付録

| ドキュメント                      | 内容             | 深度 | 確信度 | 最終更新 |
| --------------------------------- | ---------------- | ---- | ------ | -------- |
| `docs/src/glossary.md`            | 20用語。PagedAttention、KV Transfer、LMCache、Multimodal、mm_cache、Unified Compute Model等 | [SHALLOW] | [INFERRED] | 2026-02-11 |
| `.state/questions.md`             | 未解決の疑問     | -    | -      | 2026-02-11 |
| `docs/src/appendix/file-index.md` | 主要ファイル索引 | -    | -      | -        |
| `.state/reading-guide.md`         | 構造ルール6つ + ユーザー優先度。v1優先、gemma3リファレンス、CUDA中心 | [INFERRED] | [INFERRED] | 2026-02-09 |
