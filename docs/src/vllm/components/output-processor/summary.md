# OutputProcessor

> **深度**: [SHALLOW]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-11

## 概要

OutputProcessorは**フロントエンドプロセス**で動作し、バックエンド（EngineCore）からZMQ経由で受信した`EngineCoreOutput`を、ユーザー向けの`RequestOutput`に変換する。主な処理はインクリメンタルデトークナイズ、停止文字列判定、logprobs処理である。AsyncLLMの`output_handler`バックグラウンドタスクから呼び出される。

## process_outputs() フロー

**参照**: `target/vllm/vllm/v1/engine/output_processor.py:582` (process_outputs)

```
OutputProcessor.process_outputs(engine_core_outputs)       # L582
  │
  for each engine_core_output:
    │
    ├─ req_state = request_states[req_id]                  # RequestState取得
    │   （abortされていればスキップ）
    │
    ├─ 統計情報更新                                         # L620-622
    │
    ├─ デトークナイズ + 停止文字列判定                       # L637-639
    │   stop_string = detokenizer.update(
    │       new_token_ids, stop_terminated)
    │   → トークン→テキスト変換（インクリメンタル）
    │   → 停止文字列検出時は finish_reason = STOP
    │
    ├─ logprobs処理                                         # L646
    │   logprobs_processor.update_from_output(output)
    │
    ├─ RequestOutput構築                                    # L649-656
    │   req_state.make_request_output(
    │       new_token_ids, finish_reason, stop_reason, ...)
    │   → CompletionOutput + RequestOutput
    │
    ├─ 出力配信                                             # L660-665
    │   ├─ AsyncLLM: req_state.queue.put(request_output)
    │   └─ LLM: request_outputs.append(request_output)
    │
    └─ 完了処理                                             # L668-687
        if finish_reason is not None:
          _finish_request(req_state)
          → リクエスト解放、統計記録
```

## RequestState

**参照**: `target/vllm/vllm/v1/engine/output_processor.py:116` (RequestState)

各リクエストのフロントエンド側状態を保持する。OutputProcessor.add_request()で作成される。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `external_req_id` | `str` | 外部リクエストID（クライアント向け） |
| `detokenizer` | `IncrementalDetokenizer` | デトークナイザインスタンス |
| `logprobs_processor` | `LogprobsProcessor` | logprobs処理インスタンス |
| `output_kind` | `RequestOutputKind` | 出力モード（CUMULATIVE/DELTA/FINAL_ONLY） |
| `queue` | `RequestOutputCollector \| None` | AsyncLLM用出力キュー |
| `prompt_token_ids` | `list[int]` | プロンプトトークン（出力に含める用） |

### make_request_output()

**参照**: `target/vllm/vllm/v1/engine/output_processor.py:269` (make_request_output)

```
make_request_output(new_token_ids, finish_reason, ...)
  │
  ├─ FINAL_ONLY モードかつ未完了 → None（出力なし）
  │
  ├─ プーリングモデル → PoolingRequestOutput
  │
  └─ テキスト生成 → RequestOutput
      ├─ _new_completion_output()                          # L377
      │   ├─ detokenizer.get_next_output_text(finished, delta)
      │   │   → DELTAモード: 新規テキストのみ
      │   │   → CUMULATIVEモード: 全テキスト
      │   └─ CompletionOutput(text, token_ids, logprobs, ...)
      └─ RequestOutput(request_id, outputs, finished, ...)
```

## Detokenizer（インクリメンタルデトークナイズ）

**参照**: `target/vllm/vllm/v1/engine/detokenizer.py:30` (IncrementalDetokenizer)

### クラス階層

```
IncrementalDetokenizer (基底・No-op)               L30
└── BaseIncrementalDetokenizer (ABC)                L65
    ├── FastIncrementalDetokenizer                  L169
    │   → HF tokenizersの DecodeStream 使用
    └── SlowIncrementalDetokenizer                  L258
        → detokenize_incrementally() 使用
```

ファクトリメソッド `from_new_request()` がトークナイザの種類に応じて適切な実装を選択する。

### update() メソッド

**参照**: `target/vllm/vllm/v1/engine/detokenizer.py:65` (BaseIncrementalDetokenizer.update)

```
update(new_token_ids, stop_terminated) → stop_string | None
  │
  for each new_token_id:
    ├─ token_ids.append(new_token_id)
    └─ output_text += decode_next(new_token_id)  # 抽象メソッド
  │
  └─ check_stop_strings(output_text, ...)         # L316
      → (stop_string, truncate_offset) | None
```

### 停止文字列判定

**参照**: `target/vllm/vllm/v1/engine/detokenizer.py:316` (check_stop_strings)

`check_stop_strings()`は累積テキストの末尾付近で停止文字列を検索する。検出時はテキストをトランケートし、停止文字列と切り詰め位置を返す。`include_stop_str_in_output`フラグで停止文字列を出力に含めるか制御する。

## LogprobsProcessor

**参照**: `target/vllm/vllm/v1/engine/logprobs.py:28` (LogprobsProcessor)

`SamplingParams.logprobs` / `prompt_logprobs` の設定に基づいて初期化される。`update_from_output()`で`EngineCoreOutput`からlogprobs情報を抽出し、累積対数確率を更新する。

## 出力モード（RequestOutputKind）

**参照**: `target/vllm/vllm/sampling_params.py:108` (RequestOutputKind)

| モード | 値 | 動作 | 用途 |
|--------|---|------|------|
| `CUMULATIVE` | 0 | 毎回全出力テキスト/トークンを返す | デフォルト |
| `DELTA` | 1 | 差分（新規テキスト/トークン）のみ返す | ストリーミング |
| `FINAL_ONLY` | 2 | 完了時のみ出力を返す | バッチ処理 |

## AsyncLLMとの連携

```
AsyncLLM._run_output_handler()                     # async_llm.py:662
  while True:
    outputs = await engine_core.get_output_async()  # ZMQ受信
    for chunk in outputs.outputs:
      output_processor.process_outputs(chunk, ...)  # ← ここで呼ばれる
      → RequestOutputがper-requestキューにpush
      → generate()がキューからyield
```

## 上流・下流の関係

- **上流**: AsyncLLM（output_handlerタスクから呼び出し）、EngineCoreOutputs（ZMQ経由受信）
- **下流**: APIサーバー（RequestOutputをyield）

## Phase 2 深堀り候補

- `RequestOutputCollector`のキューイング実装
- ストリーミングモード（DELTA）時のテキスト差分計算詳細
- n>1サンプリング時のParentRequest管理

## 主要ファイル

| ファイル | 主要クラス/関数 |
|---------|----------------|
| `target/vllm/vllm/v1/engine/output_processor.py` | `OutputProcessor` (L73), `process_outputs()` (L582), `RequestState` (L116), `make_request_output()` (L269) |
| `target/vllm/vllm/v1/engine/detokenizer.py` | `IncrementalDetokenizer` (L30), `FastIncrementalDetokenizer` (L169), `SlowIncrementalDetokenizer` (L258), `check_stop_strings()` (L316) |
| `target/vllm/vllm/v1/engine/logprobs.py` | `LogprobsProcessor` (L28) |
| `target/vllm/vllm/outputs.py` | `RequestOutput` (L86), `CompletionOutput` (L23) |
| `target/vllm/vllm/sampling_params.py` | `RequestOutputKind` (L108) |
