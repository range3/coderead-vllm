# Phase 1c: 下流パス追跡

> **日付**: 2026-02-11
> **Phase**: 1 (垂直スライス、セッション3/3)

## 目標

テキスト推論の下流パス（Executor → Worker → GPUModelRunner → OutputProcessor）を追跡し、Phase 1の垂直スライスを完成させる。

## 主な発見

### Executor層: collective_rpc委譲パターン
- Executor抽象クラスが`collective_rpc()`で全Workerに同一メソッドを実行
- 3実装: UniProcExecutor（単一GPU）、MultiprocExecutor（TP/PP）、RayDistributedExecutor（分散）
- `execute_model()`は`collective_rpc("execute_model")`の薄いラッパー、`output[0]`のみ返す
- Worker(WorkerBase)はGPUModelRunnerへの委譲 + PP対応

### GPUModelRunner: 2フェーズ実行パターン
- **execute_model()** (L3312): 入力準備→モデルフォワード→logits計算→ExecuteModelStateに保存→None返却
- **sample_tokens()** (L3621): 状態復元→grammar適用→サンプリング→ModelRunnerOutput構築
- 分離の目的: モデルフォワード中にgrammar bitmask計算を並行実行可能にするため
- ExecuteModelState (L313): GPUテンソル（logits, hidden_states等）を保持するNamedTuple
- 6300行の理由: 10+の責務（バッチ管理、入力準備、Attention、フォワード、サンプリング、KV Transfer、SpecDec、PP、LoRA、マルチモーダル）を集約

### OutputProcessor: フロントエンドプロセスでの出力変換
- **重要**: OutputProcessorはバックエンド（EngineCore）ではなくフロントエンドプロセスで動作
- process_outputs()でEngineCoreOutput → RequestOutput変換
- Detokenizer: FastIncrementalDetokenizer（HF DecodeStream）/ SlowIncrementalDetokenizer
- インクリメンタルデトークナイズ + 停止文字列判定（check_stop_strings）
- RequestOutputKind: CUMULATIVE（デフォルト）/ DELTA（ストリーミング）/ FINAL_ONLY

### Prefill vs Decode: Unified Compute Model
- Prefill/Decode明示区別なし。num_computed_tokensの進捗で暗黙的に区分
- 同一バッチ内にPrefill/Decode混在可能（Continuous Batching）

## 成果物

- `docs/src/architecture/data-flow.md` — 下流パス、Prefill vs Decode、RequestOutput、コンポーネント優先度確定を追加
- `docs/src/components/executor/summary.md` — 新規作成 [SHALLOW]
- `docs/src/components/gpu-model-runner/summary.md` — 新規作成 [SHALLOW]
- `docs/src/components/output-processor/summary.md` — 新規作成 [SHALLOW]

## Phase 1 完了状態

テキスト推論のエンドツーエンドフローを完全に追跡完了:
1. **上流**: AsyncLLM → InputProcessor → EngineCoreClient → EngineCore
2. **コア**: EngineCore.step() → Scheduler.schedule() → KVCacheManager
3. **下流**: Executor → Worker → GPUModelRunner → OutputProcessor → RequestOutput

全コンポーネントの summary.md が作成され、data-flow.md にフロー全体が記載された。
コンポーネント優先度が確定し、Phase 2（KVCacheManager深堀り）の準備が整った。

## 次回

Phase 2 開始: KVCacheManager 深堀り（ユーザー関心1位）
