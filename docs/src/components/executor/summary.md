# Executor

> **深度**: [SHALLOW]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-11

## 概要

Executorは、EngineCoreとWorker（GPUModelRunner）の間に位置する実行委譲レイヤーである。`collective_rpc()`パターンで全Workerに対して同一メソッドを呼び出し、出力ランクのWorkerの結果を返す。単一プロセス、マルチプロセス、Ray分散の3つの実装を持つ。

## クラス階層

```
Executor (ABC)                                     abstract.py:36
├── UniProcExecutor                                uniproc_executor.py:26
│   └── ExecutorWithExternalLauncher               uniproc_executor.py:140
├── MultiprocExecutor                              multiproc_executor.py:93
└── RayDistributedExecutor                         ray_executor.py:62
```

**参照**: `target/vllm/vllm/v1/executor/abstract.py:36` (Executor)

## 主要メソッド

### collective_rpc()

**参照**: `target/vllm/vllm/v1/executor/abstract.py:180` (collective_rpc)

全Workerに対して同一メソッドを実行するRPCメカニズム。

```python
def collective_rpc(
    self,
    method: str | Callable[..., _R],  # メソッド名または関数
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    non_block: bool = False,          # True: Future返却
) -> list[_R] | Future[list[_R]]
```

### execute_model()

**参照**: `target/vllm/vllm/v1/executor/abstract.py:202` (execute_model)

```python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    non_block: bool = False,
) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
    output = self.collective_rpc("execute_model",
                                  args=(scheduler_output,),
                                  non_block=non_block)
    return output[0]   # 出力ランクWorkerの結果のみ返す
```

### sample_tokens()

**参照**: `target/vllm/vllm/v1/executor/abstract.py:222` (sample_tokens)

```python
def sample_tokens(
    self,
    grammar_output: GrammarOutput | None,
    non_block: bool = False,
) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
    output = self.collective_rpc("sample_tokens",
                                  args=(grammar_output,),
                                  non_block=non_block)
    return output[0]
```

## 実装の使い分け

| 実装 | 用途 | Worker配置 | 特徴 |
|------|------|-----------|------|
| `UniProcExecutor` | 単一GPU | ドライバプロセス内 | 最小オーバーヘッド。`max_concurrent_batches > 1`時はThreadPoolExecutor使用 |
| `MultiprocExecutor` | 複数GPU（TP/PP） | 子プロセス | MessageQueue（共有メモリ）ベース。Pipeline Parallelism対応 |
| `RayDistributedExecutor` | 分散クラスタ | Rayアクター | Ray経由のリモートWorker管理 |

## Worker（委譲先）

**参照**: `target/vllm/vllm/v1/worker/gpu_worker.py:70` (Worker)

`Worker(WorkerBase)` はGPUModelRunnerのラッパーで、以下の追加処理を行う:

- **Pipeline Parallelism**: 前段ランクからの`IntermediateTensors`受信、後段への送信
- **推論モード管理**: `@torch.inference_mode()` デコレータ

```
Worker.execute_model(scheduler_output)                    # L604
  ├─ PP: recv_tensor_dict() → IntermediateTensors        # L614-641
  ├─ model_runner.execute_model(scheduler_output, ...)    # L652
  │   → ModelRunnerOutput | None | IntermediateTensors
  ├─ PP: send_tensor_dict(intermediate_tensors)           # L660-671
  └─ return ModelRunnerOutput | None
```

## EngineCore → 出力 の委譲フロー

```
EngineCore.step()
  └─ executor.execute_model(scheduler_output, non_block=True)
      └─ collective_rpc("execute_model")
          └─ Worker.execute_model()
              └─ GPUModelRunner.execute_model()
                  → ExecuteModelState 保存、None 返却

EngineCore.step()（続き）
  └─ executor.sample_tokens(grammar_output)
      └─ collective_rpc("sample_tokens")
          └─ Worker.sample_tokens()
              └─ GPUModelRunner.sample_tokens()
                  → ModelRunnerOutput 返却
```

## 上流・下流の関係

- **上流**: EngineCore（`step()`から呼び出し）
- **下流**: Worker → GPUModelRunner

## Phase 2 深堀り候補

- MultiprocExecutorのMessageQueue実装詳細
- Pipeline Parallelism時のバッチスケジューリング
- Ray分散実行のオーバーヘッドと障害回復

## 主要ファイル

| ファイル | 主要クラス/関数 |
|---------|----------------|
| `target/vllm/vllm/v1/executor/abstract.py` | `Executor`, `collective_rpc()` (L180), `execute_model()` (L202) |
| `target/vllm/vllm/v1/executor/uniproc_executor.py` | `UniProcExecutor` (L26) |
| `target/vllm/vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor` (L93) |
| `target/vllm/vllm/v1/executor/ray_executor.py` | `RayDistributedExecutor` (L62) |
| `target/vllm/vllm/v1/worker/gpu_worker.py` | `Worker` (L70), `execute_model()` (L604) |
| `target/vllm/vllm/v1/worker/worker_base.py` | `WorkerBase` (L34), `WorkerWrapperBase` (L175) |
