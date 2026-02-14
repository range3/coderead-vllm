# Phase 2e: プロセスアーキテクチャ調査（TP=2構成）

**日付**: 2026-02-14
**Phase**: 2e
**トピック**: vLLM TP=2 プロセス構成・プロセス間通信

## セッション概要

GPU2枚・TP=2構成でのvLLMプロセスアーキテクチャを調査。4プロセス構成、3種の通信メカニズム（ZMQ/SharedMemory MQ/NCCL）、通信方式選択の設計判断を明らかにした。

## 主要な発見

### プロセス構成（4プロセス）
- **Frontend**: AsyncLLM, InputProcessor, OutputProcessor, EngineCoreClient
- **EngineCore**: EngineCore, Scheduler, KVCacheManager, MultiprocExecutor
- **VllmWorker-0**: Worker, GPUModelRunner（GPU 0, TP rank 0）
- **VllmWorker-1**: Worker, GPUModelRunner（GPU 1, TP rank 1）

### 通信メカニズム
1. **Frontend ↔ EngineCore**: ZMQ over TCP loopback（msgpack）
2. **EngineCore ↔ Workers**: SharedMemory MessageQueue（ShmRingBuffer + pickle）
3. **Worker ↔ Worker**: torch.distributed + NCCL（GPU間テンソル通信）

### ShmRingBuffer の設計
- 共有メモリ上のロックフリーリングバッファ
- メモリフェンス（~20ns）のみで同期
- 24MiB/チャンク × 10チャンク（デフォルト）
- オーバーフロー時はZMQ PUB/SUBにフォールバック
- メタデータフラグ（written + reader flags）で状態管理

### 通信方式選択の理由
- ZMQ: DP構成でのネットワーク透過性、asyncio統合
- SharedMemory MQ: 同一ノード内の低レイテンシ通信、ゼロコピー
- NCCL: GPU間テンソル通信に特化、NVLink/PCIe活用

## 成果物

- `docs/src/investigations/process-architecture.md` — 調査報告（新規作成）
- `docs/src/components/executor/summary.md` — [SHALLOW]→[MEDIUM] 昇格

## 調査したソースファイル

| ファイル | 調査箇所 |
|---------|----------|
| `target/vllm/vllm/v1/executor/multiproc_executor.py` | MultiprocExecutor全体、WorkerProc、worker_busy_loop、collective_rpc |
| `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py` | ShmRingBuffer、MessageQueue |
| `target/vllm/vllm/v1/worker/gpu_worker.py` | init_device()、torch.distributed初期化 |
| `target/vllm/vllm/v1/engine/core.py` | EngineCoreProc起動 |
| `target/vllm/vllm/v1/engine/core_client.py` | CoreEngineProcManager |

## 次回への引き継ぎ

- Pipeline Parallelism時のプロセス構成（PP=2, TP=2 → 4 Worker）は未調査
- Ray分散実行時の通信差異は未調査
- AsyncScheduling時のasync_output_busy_loop動作は未調査
