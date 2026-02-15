# Phase 2e+: SharedMemory MQ深堀り + Worker→EngineCore結果返却パス

**日付**: 2026-02-14
**Phase**: 2e+（Phase 2eの継続深堀り）

## 目的

1. SharedMemory MQ（MessageQueue/ShmRingBuffer）の内部実装詳細を調査
2. Worker→EngineCore方向の結果返却メカニズムを解明

## 調査内容

### SharedMemory MQ 深堀り

- **MessageQueueの3ロール**: Writer / Local Reader / Remote Reader
  - Writer: ShmRingBuffer + ZMQ XPUB を両方所有
  - Local Reader: ShmRingBuffer + ZMQ SUB（オーバーフロー時のみ）
  - Remote Reader: ZMQ SUB のみ（マルチノード時）

- **enqueue()バイトフォーマット**: flag(1) + buf_count(2) + [len(4)+data]×N
  - pickle protocol 5の`buffer_callback`で1MiB未満はインライン化、以上はoob buffer
  - オーバーフロー判定: total >= max_chunk_bytes (24MiB)

- **メモリフェンスプロトコル**: `threading.Lock()` acquire/release (~20ns)
  - Writer: データ書込→readerフラグリセット→**fence**→written=1→**fence**
  - Reader: **fence**→データ読み取り→read_flag=1→**fence**
  - フラグ設定順序重要（case 3経由防止）

- **SpinTimer/SpinSleepTimer**: VLLM_SLEEP_WHEN_IDLE=1で3秒後に100msスリープ移行

- **wait_until_ready()**: ZMQ XPUB/SUBでハンドシェイク（ShmRingBuffer自体にはなし）

### Worker→EngineCore 結果返却パス

- **response_mq構成**: 各WorkerがMessageQueue(1,1)を作成（writer側）。ExecutorがPipe経由でhandleを受信しreader側を構築
- **output_rankフィルタリング**: execute_model()はrank 0のみ結果返却（TP冗長排除）
- **async_scheduling**: 別スレッドでGPU→CPUコピー待ち→enqueue（メインスレッドは次RPC受信可能）
- **non_block/FutureWrapper**: Executor側の遅延評価。次回collective_rpc時にdrain

## 成果物

- `docs/src/investigations/process-architecture.md` — MessageQueue詳細セクション追加、Worker→EngineCore結果返却パスセクション追加、起動シーケンス更新、[MEDIUM]→[DEEP] 昇格

## 主要参照ファイル

- `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py` — ShmRingBuffer, MessageQueue
- `target/vllm/vllm/v1/executor/multiproc_executor.py` — MultiprocExecutor, WorkerProc
- `target/vllm/vllm/v1/outputs.py` — AsyncModelRunnerOutput
