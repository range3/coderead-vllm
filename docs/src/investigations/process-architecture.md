# プロセスアーキテクチャ（TP=2構成）

> **深度**: [MEDIUM]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-14

## 概要

vLLMをGPU2枚・TP=2で起動した場合のプロセス構成、コンポーネント配置、プロセス間通信メカニズムを調査した。

## 1. プロセス構成（合計4プロセス）

| プロセス名 | 生成元 | 含まれるコンポーネント |
|-----------|--------|----------------------|
| Frontend（メインプロセス） | ユーザー起動 | AsyncLLM, InputProcessor, EngineCoreClient, OutputProcessor |
| EngineCore (`EngineCore_DP0`) | Frontend (`mp.Process`) | EngineCore, Scheduler, KVCacheManager, MultiprocExecutor |
| VllmWorker-0 | EngineCore (`mp.Process`) | Worker, GPUModelRunner（GPU 0） |
| VllmWorker-1 | EngineCore (`mp.Process`) | Worker, GPUModelRunner（GPU 1） |

**参照**: `target/vllm/vllm/v1/engine/core_client.py:493-507` (CoreEngineProcManager)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:147-160` (WorkerProc起動)

### コンポーネントとプロセスの対応図

```
┌─ Frontend Process ─────────────────────────────────────┐
│  AsyncLLM ─→ InputProcessor                            │
│  EngineCoreClient (ZMQ ROUTER/PULL)                    │
│  OutputProcessor ←─ Detokenizer                        │
└──────────────────────────┬─────────────────────────────┘
                           │ ZMQ (msgpack)
                           ▼
┌─ EngineCore Process ─────────────────────────────────────┐
│  EngineCore.step()                                       │
│  ├─ Scheduler ─→ KVCacheManager                         │
│  └─ MultiprocExecutor                                    │
│       ├─ rpc_broadcast_mq (SharedMemory → 全Worker)      │
│       └─ worker_response_mq × 2 (各Worker → Executor)   │
└──────────┬──────────────────────────────┬────────────────┘
           │ SharedMemory MQ              │ SharedMemory MQ
           ▼                              ▼
┌─ Worker-0 Process ──┐  ┌─ Worker-1 Process ──┐
│  Worker              │  │  Worker              │
│  GPUModelRunner      │  │  GPUModelRunner      │
│  (GPU 0, TP rank 0)  │  │  (GPU 1, TP rank 1)  │
└──────────┬───────────┘  └──────────┬───────────┘
           │         NCCL            │
           └─────────────────────────┘
             (NVLink / PCIe 直接通信)
```

**注意点**:
- Scheduler、KVCacheManagerは**EngineCoreプロセス内**で動作し、独立プロセスではない
- OutputProcessorは**Frontendプロセス内**で動作する（バックエンドではない）
- MultiprocExecutorはEngineCoreプロセス内に存在し、Workerプロセスへの指令管理を行う

## 2. プロセス間通信メカニズム

### 2.1 Frontend ↔ EngineCore: ZMQ over TCP loopback

| 項目 | 値 |
|------|-----|
| プロトコル | ZMQ over TCP (`127.0.0.1:<random_port>`) |
| ソケット型 | Frontend: ROUTER(送信) + PULL(受信), EngineCore: DEALER(受信) |
| シリアライゼーション | msgpack（`msgspec.Struct(array_like)` 対応） |
| スレッドモデル | バックグラウンドスレッドでシリアライゼーション/デシリアライゼーション |

**参照**: `target/vllm/vllm/v1/engine/core_client.py:510-515` (ZMQソケット設定)
**参照**: `target/vllm/vllm/v1/engine/core.py:877-950` (EngineCoreProc._perform_handshake)

### 2.2 EngineCore ↔ Workers: SharedMemory MessageQueue

| 項目 | 値 |
|------|-----|
| プロトコル | 共有メモリ（ShmRingBuffer） + ZMQ PUB/SUB（オーバーフロー時） |
| キュー | rpc_broadcast_mq（1対多）+ worker_response_mq（各Worker→Executor） |
| シリアライゼーション | pickle（protocol 5, out-of-band buffers対応） |
| 同期方式 | ロックフリー。メモリフェンス（`threading.Lock` acquire/release, ~20ns）のみ |
| バッファサイズ | デフォルト24MiB/チャンク × 10チャンク |

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:127` (ShmRingBuffer)
**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:272` (MessageQueue)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:131-136` (rpc_broadcast_mq生成)

#### ShmRingBuffer メモリレイアウト

```
┌─────────────────────────────────┬──────────────────────────────────────┐
│ data: chunk0 | chunk1 | ... | chunkN │ metadata: [written|r0|r1|...|rN] × N │
│ max_chunks × max_chunk_bytes (24MiB) │ max_chunks × (1 + n_reader) bytes    │
└─────────────────────────────────┴──────────────────────────────────────┘
```

メタデータの状態遷移:
- `0???...???`: 未書き込み → 書き込み可
- `1000...000`: 書き込み直後 → 全reader読み取り可
- `1???...???`: 一部readerが読み取り済み
- `1111...111`: 全reader読み取り済み → 書き込み可（再利用）

**オーバーフロー処理**: データが24MiBを超える場合、ZMQ PUB/SUBソケット（IPC）経由で転送する。ローカルではXPUB/SUBソケット、リモート（マルチノード時）ではTCPソケットを使用。

#### collective_rpc の動作フロー

```
MultiprocExecutor.collective_rpc("execute_model", args=(scheduler_output,))
  │
  ├─ rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))
  │   → pickle → ShmRingBuffer書き込み → メモリフェンス
  │
  ├─ Worker-0: rpc_broadcast_mq.dequeue() → Worker.execute_model()
  │   → worker_response_mq.enqueue((SUCCESS, output))
  │
  ├─ Worker-1: rpc_broadcast_mq.dequeue() → Worker.execute_model()
  │   → worker_response_mq.enqueue((SUCCESS, output))
  │
  └─ Executor: response_mqs[0].dequeue() → output[0] を返却
      （output_rank=0 の場合、rank 0 の結果のみ返す）
```

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:303-375` (collective_rpc)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:845-871` (worker_busy_loop)

### 2.3 Worker ↔ Worker: torch.distributed + NCCL

| 項目 | 値 |
|------|-----|
| 初期化 | `torch.distributed.init_process_group(backend="nccl")` |
| Rendezvous | TCP（`tcp://127.0.0.1:<random_port>`） |
| 通信 | NCCL（NVLink / PCIe によるGPU間直接通信） |
| 用途 | Tensor Parallelの`all_reduce()`, `all_gather()`, `broadcast()` |
| タイミング | モデル forward pass 内部でレイヤーごとに実行 |

**参照**: `target/vllm/vllm/v1/worker/gpu_worker.py:263-269` (init_worker_distributed_environment)

NCCLの初期化は`Worker.init_device()`内で、メモリプロファイリング**前**に行われる。これによりNCCLバッファが確保された後の利用可能メモリが正確に計測される。

## 3. 起動シーケンス

```
1. ユーザーが AsyncLLM を生成
2. AsyncLLM → EngineCoreClient.make_async_mp_client()
3.   └─ mp.Process(target=EngineCoreProc.run_engine_core) 起動
4.       └─ EngineCore.__init__() 内で MultiprocExecutor 生成
5.           ├─ distributed_init_method = "tcp://127.0.0.1:<port>" 確保
6.           ├─ rpc_broadcast_mq (ShmRingBuffer) 作成
7.           └─ for rank in [0, 1]:
8.               mp.Process(target=WorkerProc.worker_main) 起動
9.                 ├─ Worker.init_device():
10.                │   └─ torch.distributed.init_process_group(backend="nccl")
11.                ├─ Worker.load_model(): モデルロード
12.                ├─ READY メッセージ送信（Pipe経由）
13.                └─ worker_busy_loop() でRPC待機開始
14.      └─ wait_until_ready(): 全Worker READY 待ち
15. Frontend ↔ EngineCore ZMQハンドシェイク完了
```

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:696` (WorkerProc.worker_main)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:752-770` (READY送信)

## 4. 通信方式の設計判断

### なぜ Frontend ↔ EngineCore は ZMQ なのか

1. **疎結合**: Data Parallelism構成では別ノードに配置される可能性がある。ZMQはネットワーク透過
2. **asyncio統合**: Frontendはasyncioイベントループ上で動作し、ZMQのasyncioポーラーと相性がよい
3. **バックグラウンドスレッドでの直列化**: msgpackシリアライゼーションをバックグラウンドスレッドで行い、GPU計算とオーバーラップ可能
4. **メッセージ順序保証**: ROUTER/DEALERソケットで確定的なメッセージ順序を保証

### なぜ EngineCore ↔ Workers は SharedMemory MQ なのか（ZMQではない理由）

1. **低レイテンシ**: 同一ノード内通信に特化。ZMQはネットワークソケット抽象であり、カーネル空間でのバッファコピーとシステムコールのオーバーヘッドがある
2. **ゼロコピー可能**: 共有メモリ上でpickleデータを直接読み書きでき、プロセス間のデータコピーが不要
3. **ロックフリー設計**: リングバッファ + メタデータフラグ + メモリフェンス（~20ns）で同期。ロック競合なし
4. **collective_rpc最適化**: 1対多ブロードキャスト（rpc_broadcast_mq）パターンにリングバッファが最適

### なぜ Worker ↔ Worker は NCCL なのか

1. **GPU間テンソル通信専用**: NCCLはGPUメモリ間の集合通信（all-reduce等）に特化した高性能ライブラリ
2. **NVLink活用**: GPU間直接通信でCPU介在なし。NVLink（最大900GB/s）やPCIe（最大64GB/s）を直接利用
3. **PyTorch統合**: モデルコード内の`torch.distributed`呼び出しと直接統合
4. **Pythonオブジェクト不可**: NCCLはテンソル転送専用であり、Pythonオブジェクト（SchedulerOutput等）の転送には使えない

### 通信方式比較

| 通信路 | 方式 | レイテンシ | 帯域幅 | 転送対象 | ネットワーク透過 |
|--------|------|----------|--------|---------|----------------|
| Frontend ↔ EngineCore | ZMQ (TCP) | ~µs | 中 | Pythonオブジェクト (msgpack) | Yes |
| EngineCore ↔ Workers | SharedMemory MQ | ~20ns同期 | 高 | Pythonオブジェクト (pickle) | No（同一ノード限定） |
| Worker ↔ Worker | NCCL | ~µs | 最高 | GPUテンソルのみ | Yes（multi-node NCCL対応） |

## 5. TP=1（単一GPU）との比較

TP=1の場合、`UniProcExecutor`が選択される:

| 項目 | TP=1 | TP=2 |
|------|------|------|
| Executor | UniProcExecutor | MultiprocExecutor |
| Workerプロセス | なし（EngineCoreプロセス内） | 2つの子プロセス |
| Worker通信 | 関数呼び出し（同一プロセス） | SharedMemory MQ |
| NCCL | 不要 | 必要（Worker間） |
| 合計プロセス数 | 2（Frontend + EngineCore） | 4（Frontend + EngineCore + Worker×2） |

**参照**: `target/vllm/vllm/v1/executor/uniproc_executor.py:26` (UniProcExecutor)

## 主要ファイル

| ファイル | 主要クラス/関数 |
|---------|----------------|
| `target/vllm/vllm/v1/engine/async_llm.py` | `AsyncLLM` — Frontendプロセスのエントリポイント |
| `target/vllm/vllm/v1/engine/core_client.py` | `EngineCoreClient` — ZMQ通信, CoreEngineProcManager |
| `target/vllm/vllm/v1/engine/core.py` | `EngineCore`, `EngineCoreProc` — EngineCoreプロセスのエントリポイント |
| `target/vllm/vllm/v1/executor/abstract.py` | `Executor` — collective_rpc(), execute_model() |
| `target/vllm/vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor`, `WorkerProc` — Worker起動, MessageQueue管理, worker_busy_loop |
| `target/vllm/vllm/v1/executor/uniproc_executor.py` | `UniProcExecutor` — 単一GPU用 |
| `target/vllm/vllm/v1/worker/gpu_worker.py` | `Worker` — init_device(), torch.distributed初期化 |
| `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py` | `ShmRingBuffer`, `MessageQueue` — 共有メモリ通信基盤 |
