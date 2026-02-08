# EngineCoreClient サマリー

> **深度**: [SHALLOW]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-09

## 概要

`EngineCoreClient`はフロントエンドプロセス（AsyncLLM / LLM）とバックエンドプロセス（EngineCore）間のプロセス間通信を抽象化するコンポーネントである。ZeroMQソケットとmsgpackシリアライゼーションを使用し、`EngineCoreRequest`の送信と`EngineCoreOutputs`の受信を効率的に行う。

## アーキテクチャ

```
フロントエンドプロセス            バックエンドプロセス
┌───────────────────┐            ┌───────────────────┐
│  AsyncMPClient    │            │  EngineCore       │
│                   │            │                   │
│  input_socket     ├──ROUTER──→│  (ZMQ受信)        │
│  (zmq.ROUTER)     │  msgpack   │                   │
│                   │            │                   │
│  output_socket    │←──PULL────┤  (ZMQ送信)        │
│  (zmq.PULL)       │  msgpack   │                   │
│                   │            │                   │
│  outputs_queue    │            │                   │
│  (asyncio.Queue)  │            │                   │
└───────────────────┘            └───────────────────┘
```

## 主要コンポーネント

| コンポーネント | 用途 | ファイル |
|--------------|------|---------|
| `EngineCoreClient` (ABC) | 抽象インターフェース | `target/vllm/vllm/v1/engine/core_client.py:63` |
| `MPClient` | マルチプロセスクライアント基底 | `target/vllm/vllm/v1/engine/core_client.py:442` |
| `AsyncMPClient` | 非同期マルチプロセスクライアント（AsyncLLM用） | `target/vllm/vllm/v1/engine/core_client.py:822` |
| `SyncMPClient` | 同期マルチプロセスクライアント（LLM用） | `target/vllm/vllm/v1/engine/core_client.py` |
| `DPAsyncMPClient` | データ並列（外部LB） | `target/vllm/vllm/v1/engine/core_client.py` |
| `DPLBAsyncMPClient` | データ並列（内部LB） | `target/vllm/vllm/v1/engine/core_client.py` |
| `MsgpackEncoder` | リクエストのシリアライズ | `target/vllm/vllm/v1/serial_utils.py` |
| `MsgpackDecoder` | レスポンスのデシリアライズ | `target/vllm/vllm/v1/serial_utils.py` |

## 主要メソッド

### EngineCoreClient (ABC)

| メソッド | 説明 |
|---------|------|
| `make_client()` | ファクトリ。設定に応じた適切なサブクラスを返す |
| `make_async_mp_client()` | AsyncLLM用ファクトリ。DP構成も考慮 |
| `add_request()` | EngineCoreRequestを送信 |
| `get_output()` | EngineCoreOutputsを受信 |
| `abort_requests()` | リクエストキャンセル |

### AsyncMPClient

| メソッド | 行 | 説明 |
|---------|-----|------|
| `_ensure_output_queue_task()` | L856 | ZMQ出力受信タスクを起動 |
| `get_output_async()` | L902 | asyncio.Queueから出力を取得 |
| `_send_input()` | L913 | EngineCoreRequestをZMQで送信 |
| `_send_input_message()` | L925 | ZMQ multipart送信（zero-copy対応） |

## ファクトリ選択ロジック

**参照**: `target/vllm/vllm/v1/engine/core_client.py:99` (make_async_mp_client)

```
make_async_mp_client(vllm_config, executor_class, ...)
  ├─ data_parallel_size > 1 の場合:
  │   ├─ external_lb → DPAsyncMPClient
  │   └─ internal_lb → DPLBAsyncMPClient
  └─ それ以外 → AsyncMPClient
```

## 設定

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `parallel_config.data_parallel_size` | 1 | データ並列数。>1でDP系クライアントを使用 |
| `parallel_config.data_parallel_external_lb` | — | 外部ロードバランサ使用フラグ |

## 呼び出しフロー

```
[送信パス]
AsyncLLM.add_request()
  → engine_core.add_request_async(request)
    → AsyncMPClient._send_input(REQUEST, request)
      → MsgpackEncoder.encode(request)
      → input_socket.send_multipart(msg, copy=False)
        → ZMQ ROUTER → バックエンドプロセス

[受信パス]
process_outputs_socket() [バックグラウンドタスク]
  → output_socket.recv_multipart()
    → MsgpackDecoder.decode(frames) → EngineCoreOutputs
    → outputs_queue.put_nowait(outputs)

AsyncLLM._run_output_handler()
  → engine_core.get_output_async()
    → outputs_queue.get() → EngineCoreOutputs
```

## 設計上の特徴

- **プロセス分離**: EngineCoreが別プロセスで動作するため、GILの影響を受けずスケジューリングとGPU実行を並行可能
- **msgpackシリアライゼーション**: `msgspec.Struct`の`array_like`形式でコンパクトなバイナリ表現
- **zero-copy**: ZMQ `copy=False` でメモリコピーを最小化。テンソルバッキングバッファの追跡（`add_pending_message`）
- **weakref**: 出力タスクがクライアントへの循環参照を持たないよう`weakref`を使用

## 関連ドキュメント

- [エントリポイント](../entrypoint/summary.md)
- [データフロー](../../architecture/data-flow.md)
