# プロセスアーキテクチャ（TP=2構成）

> **深度**: [DEEP]
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

#### MessageQueue の詳細設計 [DEEP] [VERIFIED]

MessageQueueは`ShmRingBuffer`をラップし、pickle protocol 5のout-of-bandバッファ対応のシリアライゼーション層を提供する。

**ロール分離（Writer / Local Reader / Remote Reader）**:

| ロール | 判定条件 | 通信手段 |
|--------|---------|----------|
| Writer | コンストラクタで生成した側 | ShmRingBuffer + ZMQ XPUB |
| Local Reader | `rank in handle.local_reader_ranks` | ShmRingBuffer + ZMQ SUB |
| Remote Reader | 上記以外 | ZMQ SUB のみ |

Writer側のコンストラクタでShmRingBufferとZMQソケット（XPUB）を両方作成する。Local Readerは共有メモリ経由で受信し、オーバーフロー時のみZMQ SUBにフォールバック。Remote Readerは常にZMQ SUBのみで受信する。

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:272-354` (MessageQueue.__init__ / create_from_handle)

**enqueue() のデータフォーマット**:

```
ShmRingBuffer チャンク内のバイトレイアウト:
+------+-------------------+--------------------+--------------------+-----+
| [0]  | [1:3]             | [3:7] [7:7+L0]     | [7+L0:11+L0] ...  | ... |
| flag | buf_count (2byte) | len0+main_pickle   | len1+oob_buffer1   | ... |
+------+-------------------+--------------------+--------------------+-----+
  flag: 0=通常, 1=オーバーフロー（ZMQ経由で後続送信）
```

- **pickle protocol 5 + out-of-band buffers**: `buffer_callback`でサイズ判定。1MiB未満のバッファはインライン化（main pickle内に含む）、1MiB以上はoob bufferとして別管理
- **オーバーフロー判定**: `total_bytes + len(main_pickle) >= max_chunk_bytes`（デフォルト24MiB）の場合、ShmRingBufferにはflag=1のみ書き込み、実データはZMQ `send_multipart`で送信
- Remote Readerへは常に`send_multipart`で送信（ShmRingBufferにアクセスできないため）

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:571-612` (enqueue)

**dequeue() のフロー**:

1. `acquire_read()`でShmRingBufferからチャンクを取得
2. flag=0（通常）: チャンクからbuf_count→各バッファ長→バッファを順次読み出し、`pickle.loads(main, buffers=oob_list)`でデシリアライズ
3. flag=1（オーバーフロー）: `acquire_read()`のコンテキストを**抜けてから**（readフラグ設定後）、ZMQ SUBソケット経由で`recv_multipart`

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:614-640` (dequeue)

**acquire_write() / acquire_read() の同期プロトコル**:

Writer:
1. メモリフェンスで最新のメタデータを読む
2. `written_flag=0`（未書き込み）または全readerが読み済み（`read_count == n_reader`）のチャンクを探す
3. `written_flag`を0にリセット → データ書き込み → 全readerフラグを0にリセット → **メモリフェンス** → `written_flag`を1に → **メモリフェンス**
4. フラグ設定順序が重要: 先にreaderフラグをリセット（case 1維持）→最後にwritten=1（case 2へ遷移）。逆順だとcase 3を経由し、readerが不整合なデータを読む危険

Reader:
1. メモリフェンスで最新のメタデータを読む
2. `written_flag=1`かつ自分の`read_flag=0`のチャンクを探す
3. データ読み取り → 自分の`read_flag`を1に → **メモリフェンス**

**SpinTimer / SpinSleepTimer**: Readerのスピン待ち戦略。デフォルトは`sched_yield()`（CPU譲渡）。`VLLM_SLEEP_WHEN_IDLE=1`時は3秒間アクティビティがないと100msスリープに移行し、CPU消費を削減する。

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:438-504` (acquire_write)
**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:506-569` (acquire_read)

**wait_until_ready() ハンドシェイク**:

Writer→各ReaderへZMQ `XPUB/SUB`経由でREADYメッセージを交換する集合操作。ShmRingBuffer自体にはハンドシェイクがないため、ZMQの`XPUB_VERBOSE`（全サブスクリプションメッセージ受信）を利用してReader接続完了を確認する。

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:405-436` (wait_until_ready)

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

### 2.4 Worker → EngineCore 結果返却パス [DEEP] [VERIFIED]

#### response_mq の構成

各Workerが**自分専用のwriter側MessageQueue**（`worker_response_mq`）を持ち、EngineCore側のMultiprocExecutorがそのreaderになる。rpc_broadcast_mq（1→多ブロードキャスト）とは逆方向の**多→1通信**だが、各MQは1 writer : 1 readerの構造。

```
┌─ EngineCore (MultiprocExecutor) ─────────────────────────┐
│                                                           │
│  response_mqs[0] ◄── reader ─── worker_response_mq (W0)  │
│  response_mqs[1] ◄── reader ─── worker_response_mq (W1)  │
│                                                           │
│  ※ 各MQは独立したShmRingBuffer (n_reader=1, n_local=1)    │
└───────────────────────────────────────────────────────────┘
```

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:508-509` (Worker側: `MessageQueue(1, 1)`)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:172-185` (Executor側: response_mqs構築)

#### response_mq のハンドシェイク

1. Worker側: `__init__`内で`MessageQueue(1, 1)`を生成（writer兼ShmRingBuffer所有者）
2. Worker側: READYメッセージと共に`worker_response_mq.export_handle()`をPipe経由でExecutor側に送信
3. Executor側: `wait_for_ready()`内でPipeからhandleを受信し、`MessageQueue.create_from_handle(handle, 0)`でreader側MQを構築
4. 双方: `wait_until_ready()`でZMQ XPUB/SUBハンドシェイク完了

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:757-770` (READY送信+ハンドシェイク)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:628-646` (wait_for_response_handle_ready)

#### 結果返却の詳細フロー

```
Worker.worker_busy_loop()
  │
  ├─ rpc_broadcast_mq.dequeue()  ← (method, args, kwargs, output_rank) を受信
  │
  ├─ func = getattr(self.worker, method)  ← "execute_model" 等
  │
  ├─ output = func(*args, **kwargs)  ← Worker.execute_model() 実行
  │
  ├─ if output_rank is None or self.rank == output_rank:
  │     ├─ [sync路] enqueue_output(output)
  │     │     ├─ isinstance(AsyncModelRunnerOutput) → .get_output()  ← GPU→CPU転送待ち
  │     │     ├─ isinstance(Exception) → (FAILURE, str(e))
  │     │     └─ else → (SUCCESS, output)
  │     │     └─ worker_response_mq.enqueue(result)
  │     │
  │     └─ [async路] async_output_queue.put(output)
  │           └─ async_output_busy_loop (別スレッド)
  │                 └─ enqueue_output(output)  ← 同上
  │
  └─ (output_rank != self.rank の場合は何も返さない)
```

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:845-871` (worker_busy_loop)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:814-843` (enqueue_output / handle_output / async_output_busy_loop)

#### output_rank による結果フィルタリング

`collective_rpc`の呼び出し時に`output_rank`（`unique_reply_rank`パラメータ）を指定できる:

| output_rank | Worker側の動作 | Executor側の動作 |
|-------------|---------------|-----------------|
| `None` | **全Workerが**結果をenqueue | **全response_mqsから**dequeue → リスト返却 |
| `0` | **rank 0のみ**enqueue | **response_mqs[0]のみ**dequeue → 単一値返却 |
| `N` | **rank Nのみ**enqueue | **response_mqs[N]のみ**dequeue → 単一値返却 |

`execute_model()`は`unique_reply_rank=self.output_rank`（通常rank 0）で呼ばれるため、**rank 0のWorkerのみが結果を返し**、他のWorkerは結果を破棄する。これはTPモデルでは全Workerが同一の出力を計算するため、1つだけ返せば十分なため。

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:270-275` (execute_model → unique_reply_rank)
**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:339-341` (response_mqs フィルタリング)

#### 非同期スケジューリング（async_scheduling）

`scheduler_config.async_scheduling=True`の場合、結果返却が非同期化される:

1. `worker_busy_loop`内の`handle_output()`が`async_output_queue`（`queue.Queue`）に出力を投入
2. 別スレッド`async_output_busy_loop`（デーモンスレッド `WorkerAsyncOutputCopy`）がキューから取り出し
3. `AsyncModelRunnerOutput.get_output()`でGPU→CPU非同期コピー完了を待機
4. `worker_response_mq.enqueue()`で結果をEngineCore側に送信

これにより、worker_busy_loopスレッドは**GPU→CPUコピー完了を待たずに次のRPCを受信**できる。GPU計算と結果転送をパイプライン化する仕組み。

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:560-568` (async_output_copy_thread起動)
**参照**: `target/vllm/vllm/v1/outputs.py:200-209` (AsyncModelRunnerOutput)

#### non_block（FutureWrapper）

Executor側の`collective_rpc(non_block=True)`では、response_mqからの結果取得を**遅延評価**する:

1. `get_response`クロージャを`FutureWrapper`に包んで即座に返す
2. 次回の`collective_rpc`呼び出し時に、pending futuresを先にdrainする（`futures_queue`から順次pop→`wait_for_response`）
3. 実際にresponse_mqから`dequeue()`するのはdrain時

これにより、Executor側も結果待ちなしで次のRPCブロードキャストを発行でき、EngineCore.step()内のスケジューリングとWorkerの計算をオーバーラップできる。

**参照**: `target/vllm/vllm/v1/executor/multiproc_executor.py:365-375` (non_block / FutureWrapper)

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
1.  ユーザーが AsyncLLM を生成
2.  AsyncLLM → EngineCoreClient.make_async_mp_client()
3.    └─ mp.Process(target=EngineCoreProc.run_engine_core) 起動
4.        └─ EngineCore.__init__() 内で MultiprocExecutor 生成
5.            ├─ distributed_init_method = "tcp://127.0.0.1:<port>" 確保
6.            ├─ rpc_broadcast_mq (ShmRingBuffer, n_reader=2) 作成
7.            └─ for rank in [0, 1]:
8.                mp.Process(target=WorkerProc.worker_main) 起動
9.                  ├─ Worker.init_device():
10.                 │   └─ torch.distributed.init_process_group(backend="nccl")
11.                 ├─ Worker.load_model(): モデルロード
12.                 ├─ _init_message_queues():
13.                 │   ├─ rpc_broadcast_mq = create_from_handle(input_shm_handle, rank)
14.                 │   └─ worker_response_mq = MessageQueue(1, 1)  ← 各Worker独自
15.                 ├─ READY メッセージ + response_mq handle 送信（Pipe経由）
16.                 └─ wait_until_ready() → worker_busy_loop() でRPC待機開始
17.       └─ wait_for_ready():
18.            ├─ Pipeからhandle受信 → response_mqs[rank] 構築
19.            ├─ rpc_broadcast_mq.wait_until_ready()
20.            └─ 各response_mq.wait_until_ready()
21. Frontend ↔ EngineCore ZMQハンドシェイク完了
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
