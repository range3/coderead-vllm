# テキスト推論データフロー

> **深度**: [MEDIUM]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-11

## 概要

テキスト推論リクエストは、APIエントリポイントからエンジン層を経てGPUで実行され、生成されたトークンがデトークナイズされてユーザーに返却される。フロー全体は5つの境界データ構造（EngineCoreRequest → SchedulerOutput → ModelRunnerOutput → EngineCoreOutput → RequestOutput）で区切られ、ZMQ IPCによるプロセス分離とasyncioによる非同期パイプラインで高スループットを実現する。

## フロー全体図

```mermaid
graph TD
    subgraph フロントエンドプロセス
        API["API Server / LLM"]
        AsyncLLM["AsyncLLM<br>generate() / add_request()"]
        IP["InputProcessor<br>process_inputs()"]
        OP["OutputProcessor<br>process_outputs()"]
        Client["EngineCoreClient<br>AsyncMPClient"]
    end

    subgraph バックエンドプロセス ["EngineCore プロセス"]
        EC["EngineCore<br>step()"]
        Sched["Scheduler<br>schedule()"]
        KV["KVCacheManager<br>allocate_slots()"]
        Exec["Executor<br>execute_model()"]
        Worker["Worker"]
        MR["GPUModelRunner<br>execute_model()"]
    end

    API -->|"prompt, params"| AsyncLLM
    AsyncLLM -->|"prompt, params"| IP
    IP -->|"EngineCoreRequest"| AsyncLLM
    AsyncLLM -->|"EngineCoreRequest"| Client

    Client -->|"ZMQ ROUTER\nmsgpack"| EC
    EC --> Sched
    Sched -->|"allocate_slots()"| KV
    Sched -->|"SchedulerOutput"| EC
    EC -->|"SchedulerOutput"| Exec
    Exec -->|"MessageQueue\n共有メモリ"| Worker
    Worker --> MR
    MR -->|"ModelRunnerOutput"| Worker
    Worker -->|"ModelRunnerOutput"| Exec
    Exec -->|"ModelRunnerOutput"| EC
    EC -->|"update_from_output()"| Sched
    Sched -->|"EngineCoreOutputs"| EC

    EC -->|"ZMQ PUSH\nmsgpack"| Client
    Client -->|"EngineCoreOutputs"| OP
    OP -->|"RequestOutput"| AsyncLLM
    AsyncLLM -->|"RequestOutput"| API
```

## 境界データ構造

フローは以下の5つのデータ構造で区切られる。各構造はプロセス間またはコンポーネント間の境界を定義する。

### EngineCoreRequest

フロントエンド → バックエンドの境界。ユーザー入力を正規化した内部表現。

**参照**: `target/vllm/vllm/v1/engine/__init__.py:55` (EngineCoreRequest)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `request_id` | `str` | 内部リクエストID（外部IDに8文字ランダムサフィックス付与） |
| `prompt_token_ids` | `list[int] \| None` | トークナイズ済みプロンプト |
| `mm_features` | `list[MultiModalFeatureSpec] \| None` | マルチモーダル入力（テキスト推論ではNone） |
| `sampling_params` | `SamplingParams \| None` | サンプリングパラメータ（clone済み） |
| `eos_token_id` | `int \| None` | 終了トークンID |
| `arrival_time` | `float` | リクエスト到着時刻 |
| `lora_request` | `LoRARequest \| None` | LoRAアダプタ情報 |
| `priority` | `int` | 優先度（デフォルト0） |
| `data_parallel_rank` | `int \| None` | データ並列ランク指定 |

`msgspec.Struct` を継承し、`array_like=True` + `omit_defaults=True` で効率的にmsgpackシリアライズされる。

### SchedulerOutput

Scheduler → Executor の境界。各ステップのスケジュール結果を含む。

**参照**: `target/vllm/vllm/v1/core/sched/output.py:184` (SchedulerOutput)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `scheduled_new_reqs` | `list[NewRequestData]` | 初回スケジュールされたリクエスト（フルデータ） |
| `scheduled_cached_reqs` | `CachedRequestData` | 既スケジュール済みリクエスト（差分のみ） |
| `num_scheduled_tokens` | `dict[str, int]` | リクエストごとのスケジュールトークン数 |
| `total_num_scheduled_tokens` | `int` | 合計スケジュールトークン数 |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | Speculative Decoding用トークン |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | エンコーダ入力インデックス（マルチモーダル） |
| `num_common_prefix_blocks` | `list[int]` | 共通プレフィックスブロック数（Cascade Attention用） |
| `finished_req_ids` | `set[str]` | このステップで完了したリクエストID |
| `free_encoder_mm_hashes` | `list[str]` | 解放するエンコーダキャッシュのmm_hash |
| `preempted_req_ids` | `set[str] \| None` | プリエンプションされたリクエスト |
| `has_structured_output_requests` | `bool` | 構造化出力リクエストの有無 |
| `pending_structured_output_tokens` | `bool` | Grammar bitmask準備状態 |
| `num_invalid_spec_tokens` | `dict[str, int] \| None` | 無効スペキュレーショントークン数 |
| `kv_connector_metadata` | `KVConnectorMetadata \| None` | KV Transfer メタデータ |
| `ec_connector_metadata` | `ECConnectorMetadata \| None` | EC Transfer メタデータ |

**NewRequestData** は初回スケジュール時のフルデータ（プロンプトトークン、サンプリングパラメータ、ブロックID等）を含む。**CachedRequestData** は既スケジュール済みリクエストの差分（新規ブロックID、新トークンID、計算済みトークン数の更新）のみを含み、プロセス間通信コストを最小化する。

### ModelRunnerOutput

GPUModelRunner → EngineCore の境界。モデル推論結果を含む。

**参照**: `target/vllm/vllm/v1/outputs.py:160` (ModelRunnerOutput)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `req_ids` | `list[str]` | バッチ内のリクエストID一覧 |
| `req_id_to_index` | `dict[str, int]` | リクエストID → バッチインデックス |
| `sampled_token_ids` | `list[list[int]]` | サンプリング済みトークンID [num_reqs, num_generated] |
| `logprobs` | `LogprobsLists \| None` | 生成トークンの対数確率 |
| `prompt_logprobs_dict` | `dict[str, LogprobsTensors \| None]` | プロンプトトークンの対数確率 |
| `pooler_output` | `list[Tensor \| None] \| None` | プーリング出力（埋め込みモデル用） |
| `kv_connector_output` | `KVConnectorOutput \| None` | KV Transfer出力 |
| `ec_connector_output` | `ECConnectorOutput \| None` | EC Transfer出力 |
| `num_nans_in_logits` | `dict[str, int] \| None` | logits内のNaN数 |
| `cudagraph_stats` | `CUDAGraphStat \| None` | CUDAGraph実行統計 |

Worker→Executorへの転送ではPythonリスト形式を使用し、torch.Tensorの高コストなシリアライゼーションを回避する。

### EngineCoreOutput

バックエンド → フロントエンドの境界。リクエスト単位の推論結果。

**参照**: `target/vllm/vllm/v1/engine/__init__.py:130` (EngineCoreOutput)

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `request_id` | `str` | 対応するリクエストID |
| `new_token_ids` | `list[int]` | 新たに生成されたトークンID |
| `finish_reason` | `FinishReason \| None` | 完了理由（stop/length/abort/error） |
| `new_logprobs` | `LogprobsLists \| None` | 生成トークンのlogprobs |
| `num_cached_tokens` | `int` | プレフィックスキャッシュヒット数 |

`EngineCoreOutputs`（複数形）がこれを`list[EngineCoreOutput]`としてバッチ化し、`scheduler_stats`やタイムスタンプと共にZMQ経由で送信される。

**参照**: `target/vllm/vllm/v1/engine/__init__.py:176` (EngineCoreOutputs)

### RequestOutput [TODO]

OutputProcessor → API の境界。ユーザーに返却される最終出力。セッション3で詳細記述。

## 上流パス: リクエスト受信 → EngineCore

### エントリポイント (LLM / AsyncLLM)

vLLMには同期パス（`LLM`）と非同期パス（`AsyncLLM`）の2つのエントリポイントがある。内部的にはどちらも`InputProcessor`と`EngineCoreClient`を使用する。

#### 非同期パス（主パス）: AsyncLLM

APIサーバー（OpenAI互換API等）が使用する主要パス。

**参照**: `target/vllm/vllm/v1/engine/async_llm.py:71` (AsyncLLM)

```
AsyncLLM.generate(prompt, sampling_params, request_id)    # L537
  │
  ├─ add_request(request_id, prompt, params)               # L286
  │   ├─ input_processor.process_inputs(prompt, params)    # L364
  │   │   → EngineCoreRequest を生成
  │   ├─ input_processor.assign_request_id(request)        # L378
  │   │   → 内部IDを付与（外部ID + 8文字ランダムサフィックス）
  │   ├─ output_processor.add_request(request, ...)        # L423
  │   │   → フロントエンド側でリクエストを登録
  │   └─ engine_core.add_request_async(request)            # L426
  │       → ZMQ経由でバックエンドへ送信
  │
  └─ while not finished:                                    # L586
      out = q.get_nowait() or await q.get()                # L589
      yield out                                             # L596
```

`generate()`はAsyncGeneratorで、バックグラウンドの`output_handler`タスクがEngineCoreからの出力を`RequestOutputCollector`キューにpushし、`generate()`がそれをyieldする。

**output_handler（バックグラウンドタスク）**:

**参照**: `target/vllm/vllm/v1/engine/async_llm.py:647` (_run_output_handler)

```
output_handler():                                           # L662
  while True:
    outputs = await engine_core.get_output_async()          # L666
    for chunk in outputs.outputs:                           # L677
      output_processor.process_outputs(chunk, ...)          # L681
      → RequestOutputをキューにpush
    if reqs_to_abort:
      await engine_core.abort_requests_async(...)           # L693
```

#### 同期パス: LLM

オフライン推論（バッチ処理）で使用される。

**参照**: `target/vllm/vllm/entrypoints/llm.py:396` (generate)

```
LLM.generate(prompts, sampling_params)                      # L396
  → _run_completion(prompts, params)                        # L449
    → _add_request(prompt, params) × N                      # L1850
    │  ├─ input_processor.process_inputs(prompt, params)    # L1879
    │  └─ llm_engine.add_request(request_id, request, ...)  # L1889
    → _run_engine()                                         # L1900
       while has_unfinished_requests():                     # L1918
         step_outputs = llm_engine.step()                   # L1919
```

同期パスとの主な違い:
- `LLM`は`_run_engine()`でポーリングループを回す（AsyncGeneratorではない）
- `llm_engine`（=`AsyncLLM`のラッパー）の`step()`を直接呼ぶ
- プログレスバー（tqdm）でバッチ処理の進捗を表示

### 入力処理 (InputProcessor)

`InputProcessor`はユーザー入力（テキストプロンプト、パラメータ）を`EngineCoreRequest`に変換する。

**参照**: `target/vllm/vllm/v1/engine/input_processor.py:56` (InputProcessor)

```
InputProcessor.process_inputs(request_id, prompt, params)   # L521
  ├─ _validate_lora(lora_request)                           # L535
  ├─ _validate_params(params)                               # L536
  ├─ data_parallel_rank の範囲チェック                       # L542
  ├─ arrival_time 設定（未指定なら time.time()）             # L548
  │
  ├─ input_preprocessor.preprocess(prompt, ...)             # L581
  │   → テキストをトークナイズ（tokenizer.encode()）
  │   → ProcessorInputs を返す
  │
  ├─ split_enc_dec_inputs(processed_inputs)                 # L597
  │   → エンコーダ/デコーダ入力を分離
  │
  ├─ SamplingParams の正規化                                # L608-623
  │   ├─ params.clone()                                     # L612
  │   ├─ max_tokens 未設定時: max_model_len - seq_len       # L614-618
  │   ├─ update_from_generation_config()                    # L619
  │   └─ update_from_tokenizer()                            # L623
  │
  └─ EngineCoreRequest を構築して返す                        # L656-671
```

テキスト推論の場合、マルチモーダル関連処理（L630-654）はスキップされる（`mm_features`はNone）。

### プロセス間通信 (EngineCoreClient / ZMQ IPC)

`EngineCoreClient`はフロントエンドプロセスとバックエンドプロセス（EngineCore）間のZMQ IPC通信を担当する。

**参照**: `target/vllm/vllm/v1/engine/core_client.py:63` (EngineCoreClient)

#### クライアント階層

| クラス | 用途 | トランスポート |
|--------|------|--------------|
| `EngineCoreClient` (ABC) | 抽象インターフェース | — |
| `InprocClient` | インプロセス（デバッグ用） | 直接呼び出し |
| `SyncMPClient` | 同期マルチプロセス（LLM用） | ZMQ同期 |
| `AsyncMPClient` | 非同期マルチプロセス（AsyncLLM用） | ZMQ非同期 |
| `DPAsyncMPClient` | データ並列（外部LB） | 複数ZMQ |
| `DPLBAsyncMPClient` | データ並列（内部LB） | 複数ZMQ |

**参照**: `target/vllm/vllm/v1/engine/core_client.py:442` (MPClient)

#### ZMQソケット構成

```
フロントエンド                 バックエンド
┌──────────────┐              ┌──────────────┐
│ AsyncMPClient│              │ EngineCore   │
│              │              │              │
│ input_socket ├─── ROUTER ──→│ (受信)       │
│ (zmq.ROUTER) │    msgpack   │              │
│              │              │              │
│ output_socket│←── PULL ─────┤ (送信)       │
│ (zmq.PULL)   │    msgpack   │              │
└──────────────┘              └──────────────┘
```

- **シリアライゼーション**: `MsgpackEncoder` / `MsgpackDecoder`（`msgspec`ライブラリ）
  - `EngineCoreRequest`のシリアライズ → 入力ソケット経由で送信
  - `EngineCoreOutputs`のデシリアライズ ← 出力ソケット経由で受信
- **非同期出力受信**: `process_outputs_socket()`タスクがZMQソケットをポーリングし、受信したOutputsを`asyncio.Queue`にpush

**参照**: `target/vllm/vllm/v1/engine/core_client.py:822` (AsyncMPClient)

#### EngineCore側のリクエスト受信

`EngineCore.add_request()`はリクエストをバリデーションしてSchedulerに登録する。

**参照**: `target/vllm/vllm/v1/engine/core.py:288` (add_request)

```
EngineCore.add_request(request)                             # L288
  ├─ request_id の型チェック                                 # L295
  ├─ pooling_params のバリデーション                          # L300
  ├─ kv_transfer_params の互換性チェック                      # L311
  └─ scheduler.add_request(request)                          # L319
```

#### EngineCore.step() （コアループ概要）

**参照**: `target/vllm/vllm/v1/engine/core.py:389` (step)

```
EngineCore.step()                                            # L389
  ├─ scheduler.schedule()        → SchedulerOutput           # L404
  ├─ executor.execute_model()    → Future[ModelRunnerOutput]  # L405
  ├─ grammar_output 取得                                      # L406
  ├─ future.result()             → ModelRunnerOutput          # L411
  ├─ sample_tokens()（非同期スケジューリング時）              # L413
  └─ scheduler.update_from_output() → EngineCoreOutputs      # L418
```

## コアループ: EngineCore.step()

EngineCoreの`step()`メソッドは、各ステップで **schedule → execute → update** のサイクルを実行し、待機中のリクエストから生成トークンを生産する。

### step() 実行フロー

**参照**: `target/vllm/vllm/v1/engine/core.py:389` (step)

```
EngineCore.step()                                            # L389
  │
  ├─ if _scheduler_paused: return {}, False                  # L397
  ├─ if not scheduler.has_requests(): return {}, False       # L402
  │
  ├─ 1. scheduler_output = scheduler.schedule()              # L404
  │      → SchedulerOutput
  │      （RUNNINGリクエストの予算割当 → WAITINGリクエストの受け入れ
  │        → KVキャッシュブロック確保 → SchedulerOutput構築）
  │
  ├─ 2. future = executor.execute_model(                     # L405
  │         scheduler_output, non_block=True)
  │      → Future[ModelRunnerOutput | None]
  │      （非ブロッキング。ワーカープロセスで並行実行）
  │
  ├─ 3. grammar_output = scheduler.get_grammar_bitmask(      # L406
  │         scheduler_output)
  │      （構造化出力有効時のみ使用）
  │
  ├─ 4. model_output = future.result()                       # L411
  │      → ModelRunnerOutput（ブロッキング待機）
  │
  ├─ 5. if model_output is None:                             # L413
  │        model_output = executor.sample_tokens(grammar_output)
  │      （非同期スケジューリング時: execute_modelとsamplingが分離）
  │
  ├─ 6. _process_aborts_queue()                              # L417
  │
  └─ 7. engine_core_outputs = scheduler.update_from_output(  # L418
  │         scheduler_output, model_output)
  │      → dict[int, EngineCoreOutputs]
  │      （生成トークンの追加、完了判定、出力構築）
  │
  └─ return (engine_core_outputs,                            # L422
             total_num_scheduled_tokens > 0)
```

### Scheduler と KVCacheManager の相互作用

```mermaid
sequenceDiagram
    participant EC as EngineCore
    participant S as Scheduler
    participant KV as KVCacheManager
    participant Ex as Executor

    EC->>S: schedule()

    Note over S: Phase 1: RUNNINGリクエスト処理
    loop 各RUNNINGリクエスト
        S->>KV: allocate_slots(request, num_new_tokens)
        KV-->>S: KVCacheBlocks or None
        alt 割り当て失敗 (None)
            S->>KV: free(低優先度request)
            Note over S: プリエンプション → 再試行
        end
    end

    Note over S: Phase 2: WAITINGリクエスト受け入れ
    loop 各WAITINGリクエスト
        S->>KV: get_computed_blocks(request)
        KV-->>S: (cached_blocks, num_hits)
        S->>KV: allocate_slots(request, num_new_tokens, ...)
        KV-->>S: KVCacheBlocks or None
        alt 割り当て失敗 (None)
            Note over S: break（ループ終了）
        end
    end

    Note over S: Phase 3: SchedulerOutput構築
    S-->>EC: SchedulerOutput

    EC->>Ex: execute_model(scheduler_output)
    Ex-->>EC: Future[ModelRunnerOutput]
    EC->>EC: future.result()（待機）

    EC->>S: update_from_output(scheduler_output, model_output)
    Note over S: トークン追加、完了判定
    S-->>EC: dict[int, EngineCoreOutputs]
```

### Scheduler.schedule() の3フェーズ

**参照**: `target/vllm/vllm/v1/core/sched/scheduler.py:321` (schedule)

`schedule()` は Unified Compute Model を採用し、Prefill/Decodeを区別せず `num_computed_tokens` の進捗で統一的にトークンを割り当てる。

| フェーズ | 行 | 対象 | 処理 |
|---------|-----|------|------|
| Phase 1 | L350-517 | RUNNINGリクエスト | トークン予算割当。ブロック不足時はプリエンプション |
| Phase 2 | L532-800 | WAITINGリクエスト | 新規受け入れ。プレフィックスキャッシュ検索 + ブロック割当 |
| Phase 3 | L827-896 | 出力構築 | NewRequestData + CachedRequestData → SchedulerOutput |

**トークン予算**: `token_budget = max_num_scheduled_tokens`（ステップあたり上限）で、各リクエストのスケジュール時に消費される。

詳細は [Scheduler サマリー](../components/scheduler/summary.md) を参照。

### KVCacheManager のブロック割り当て

**参照**: `target/vllm/vllm/v1/core/kv_cache_manager.py:206` (allocate_slots)

`allocate_slots()` は以下のブロック配置に基づいてGPUメモリブロックを確保する:

```
|  comp  | new_comp | ext_comp |   new   | lookahead |
|<------ 既計算トークン ------>|<-- 新規計算対象 -->|
                               |<- 割り当て対象 ->|
```

- 成功時: `KVCacheBlocks`（割り当てたブロック情報）を返す
- 失敗時: `None` を返す → Schedulerがプリエンプション（RUNNING）またはスキップ（WAITING）

プレフィックスキャッシュ検索は `get_computed_blocks()` で行い、過去に計算済みのブロックを再利用する。

詳細は [KVCacheManager サマリー](../components/kv-cache-manager/summary.md) を参照。

### update_from_output() → EngineCoreOutputs

**参照**: `target/vllm/vllm/v1/core/sched/scheduler.py:1241` (update_from_output)

`ModelRunnerOutput`を受けてSchedulerの状態を更新し、クライアントに返す`EngineCoreOutputs`を構築する。

```
update_from_output(scheduler_output, model_runner_output)
  for each scheduled request:
    ├─ Speculative Decodingリジェクション処理
    │   → 不採用分の num_computed_tokens 巻き戻し
    ├─ 生成トークンをリクエストに追加
    ├─ 完了判定（EOS、max_tokens、stop_token）
    │   → 完了時: kv_cache_manager.free(request) でブロック解放
    └─ EngineCoreOutput 構築（request_id, new_token_ids, finish_reason, ...）
  → dict[int, EngineCoreOutputs]（クライアントインデックス別）
```

## 下流パス: ModelRunnerOutput → ユーザー応答 [TODO]

セッション3で詳細記述。GPUModelRunner、OutputProcessor、Detokenizerの処理を追跡する。

## Prefill vs Decode [TODO]

セッション3で記述。同一フロー内でのPrefill/Decodeフェーズの違いを整理する。

## コンポーネント優先度（暫定）

Phase 2での深堀り順序。セッション3で確定する。

| 優先度 | コンポーネント | 理由 |
|--------|--------------|------|
| **S** | KVCacheManager | ユーザー関心1位（メモリ管理/KVキャッシュ） |
| **A** | Scheduler | KVCacheManagerと密連携、推論パイプライン全体を制御 |
| **B** | EngineCore, AsyncLLM, GPUModelRunner | フロー理解に重要 |
| **C** | InputProcessor, EngineCoreClient, Executor, Worker, OutputProcessor | 薄いレイヤーまたは通信層 |

## 参照ファイル一覧

| ファイル | 主要クラス/関数 | 役割 |
|---------|----------------|------|
| `target/vllm/vllm/entrypoints/llm.py` | `LLM.generate()` (L396), `_add_request()` (L1850) | 同期エントリポイント |
| `target/vllm/vllm/v1/engine/async_llm.py` | `AsyncLLM.generate()` (L537), `add_request()` (L286) | 非同期エントリポイント |
| `target/vllm/vllm/v1/engine/input_processor.py` | `InputProcessor.process_inputs()` (L521) | 入力処理 |
| `target/vllm/vllm/v1/engine/__init__.py` | `EngineCoreRequest` (L55), `EngineCoreOutput` (L130), `EngineCoreOutputs` (L176) | 境界データ構造 |
| `target/vllm/vllm/v1/engine/core_client.py` | `EngineCoreClient` (L63), `MPClient` (L442), `AsyncMPClient` (L822) | ZMQ IPC通信 |
| `target/vllm/vllm/v1/engine/core.py` | `EngineCore.add_request()` (L288), `step()` (L389) | 推論ループ本体 |
| `target/vllm/vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` (L321), `update_from_output()` (L1241) | スケジューリング |
| `target/vllm/vllm/v1/core/sched/output.py` | `SchedulerOutput` (L184), `NewRequestData` (L34), `CachedRequestData` (L114) | スケジュール出力データ構造 |
| `target/vllm/vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()` (L206), `get_computed_blocks()` (L164) | KVキャッシュ管理 |
| `target/vllm/vllm/v1/core/block_pool.py` | `BlockPool` (L128) | 物理ブロック管理 |
| `target/vllm/vllm/v1/request.py` | `Request` | リクエスト内部状態 |
| `target/vllm/vllm/v1/outputs.py` | `ModelRunnerOutput` (L160) | モデル推論出力 |
| `target/vllm/vllm/v1/executor/abstract.py` | `Executor` (ABC) | 実行層抽象 |
| `target/vllm/vllm/v1/worker/gpu_worker.py` | `Worker.execute_model()` (L604) | GPU Worker |
| `target/vllm/vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model()` (L3312) | モデル実行 |
| `target/vllm/vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs()` | 出力処理 |
