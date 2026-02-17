# ZMQ 通信パターンと信頼性分析

> **深度**: [MEDIUM]
> **確信度**: [VERIFIED]
> **日付**: 2026-02-18
> **きっかけ**: vLLM全体のプロセス間通信基盤であるZMQの使用パターンを体系的に理解し、メッセージ喪失時の挙動を分析する

## 問い

1. vLLMはZMQのどのソケットタイプ・通信パターンを使っているか？
2. ZMQにはネイティブな到達保証やリトライがないが、メッセージが喪失した場合はどうなるか？
3. vLLM側で信頼性を担保する仕組みはあるか？

## ZMQ使用箇所の全体像

vLLM（v1）では16ファイルでZMQが使用されており、以下の5カテゴリに分類できる。

### カテゴリ一覧

| カテゴリ | ファイル数 | トランスポート | 用途 |
|---------|-----------|---------------|------|
| Frontend↔EngineCore通信 | 5 | IPC / TCP | コアのリクエスト/レスポンス通信 |
| DP Coordinator | 1 | IPC / TCP | Data Parallel負荷分散・Wave調整 |
| MessageQueue (ShmRingBuffer) | 1 | IPC | SharedMemoryオーバーフロー時のフォールバック |
| KV Cache Events | 1 | TCP / IPC | 外部サービスへのKVイベント配信 |
| KV Transfer コネクタ | 8 | TCP | ノード間KVキャッシュ転送の制御チャネル |

## 1. Frontend↔EngineCore通信 [VERIFIED]

最も重要な通信パス。フロントエンド（AsyncLLM/LLM）とEngineCore間のリクエスト送信・レスポンス受信を担う。

### ソケット構成

```
Frontend (MPClient)              EngineCore (EngineCoreProc)
┌─────────────────┐              ┌─────────────────┐
│  input_socket    │ ──ROUTER──► │  input_socket    │
│  (zmq.ROUTER,    │              │  (zmq.DEALER,    │
│   bind=True)     │              │   bind=False)    │
│                  │              │                  │
│  output_socket   │ ◄──PULL──── │  output_socket   │
│  (zmq.PULL,      │              │  (zmq.PUSH,      │
│   bind=False)    │              │   bind=True)     │ ← linger=4000ms
└─────────────────┘              └─────────────────┘
```

**参照**: `target/vllm/vllm/v1/engine/core_client.py:510-514` (ソケット作成)
**参照**: `target/vllm/vllm/v1/engine/core.py:1199-1206` (EngineCore側input)
**参照**: `target/vllm/vllm/v1/engine/core.py:1286-1296` (EngineCore側output)

### パターン: ROUTER/DEALER

- **リクエスト送信**: Frontend(ROUTER) → EngineCore(DEALER)
  - ROUTERはidentityベースのルーティングを行う。DPモードでは複数EngineCoreへの振り分けに使用
  - EngineCoreのidentityはDP rankの2バイトリトルエンディアン表現
  - メッセージ形式: `(identity, request_type, serialized_data, [oob_buffers...])`

- **レスポンス返却**: EngineCore(PUSH) → Frontend(PULL)
  - PUSH/PULLはunidirectionalで、identityルーティングなし
  - 複数API serverがある場合、各API serverに別々のPUSH→PULLペア
  - `client_index`で宛先のPUSHソケットを選択

### HWM（High Water Mark）設定

**参照**: `target/vllm/vllm/utils/network_utils.py:260-313` (make_zmq_socket)

```python
# PULL, DEALER, ROUTER
socket.setsockopt(zmq.RCVHWM, 0)  # 受信HWM無制限
socket.setsockopt(zmq.RCVBUF, buf_size)  # 0.5GB or system default

# PUSH, DEALER, ROUTER
socket.setsockopt(zmq.SNDHWM, 0)  # 送信HWM無制限
socket.setsockopt(zmq.SNDBUF, buf_size)  # 0.5GB or system default
```

**重要**: HWMが0（無制限）に設定されているため、**送信側でのメッセージドロップは発生しない**。ZMQはHWMに達した場合にメッセージをドロップするが、HWM=0ではカーネルバッファが許す限りキューイングされる。

### ハンドシェイクプロトコル

起動時の3段階ハンドシェイク:

1. **EngineCore→Frontend**: DEALER→ROUTERで空メッセージ`b""`送信（ROUTER側がidentityを認識するため必須）
2. **EngineCore→Frontend**: `"HELLO"`メッセージ送信（DP rank、local/remote情報）
3. **Frontend→EngineCore**: `EngineHandshakeMetadata`返却（ZMQアドレス、parallel_config）
4. **EngineCore→Frontend**: `"READY"`メッセージ送信（初期化完了、num_gpu_blocks報告）

**参照**: `target/vllm/vllm/v1/engine/utils.py:937-1091` (wait_for_engine_startup)
**参照**: `target/vllm/vllm/v1/engine/core.py:870-920` (EngineCore側ハンドシェイク)

### ゼロコピー送信とMessageTracker

メッセージにテンソルのバッキングバッファが含まれる場合、`send_multipart(copy=False, track=True)`でゼロコピー送信を行う。`zmq.MessageTracker`でZMQがバッファを使い終わるまで参照を保持する。

**参照**: `target/vllm/vllm/v1/engine/core_client.py:581-587` (pending_messages管理)
**参照**: `target/vllm/vllm/v1/engine/core.py:1322-1332` (output側のpending管理+バッファ再利用)

## 2. DP Coordinator通信 [VERIFIED]

Data Parallel環境での負荷分散統計の集約・配信とWave調整を担う。

### ソケット構成

```
Frontend(s)                DPCoordinator            EngineCore(s)
┌──────────┐              ┌──────────────┐          ┌──────────┐
│stats_upd │◄──XSUB────── │publish_front │          │          │
│(zmq.XSUB)│              │(zmq.XPUB,    │          │          │
│          │──────────────►│ bind=True)   │          │          │
│          │ 新リクエスト通知│              │          │          │
│          │              │              │          │          │
│          │              │output_back   │◄──PUSH── │coord_out │
│          │              │(zmq.PULL,    │          │(zmq.PUSH)│
│          │              │ bind=True)   │          │          │
│          │              │              │          │          │
│          │              │publish_back  │──XPUB──► │coord_in  │
│          │              │(zmq.XPUB,    │          │(zmq.XSUB)│
│          │              │ bind=True)   │          │          │
└──────────┘              └──────────────┘          └──────────┘
```

**参照**: `target/vllm/vllm/v1/engine/coordinator.py:113-395`

### 3つの通信チャネル

1. **publish_front (XPUB)**: Coordinator→Frontend。統計情報（各エンジンのwaiting/running数）とwave状態を配信
2. **output_back (PULL)**: EngineCore→Coordinator。各エンジンのScheduler統計とwave完了通知
3. **publish_back (XPUB)**: Coordinator→EngineCore。wave開始指示のブロードキャスト

### XPUB/XSUBパターン

通常のPUB/SUBと異なり、XPUB/XSUBはサブスクリプションメッセージを可視化できる:
- **XPUB**: subscription/unsubscriptionメッセージを受信可能 → 全サブスクライバの接続確認に使用
- **XSUB**: 明示的にsubscriptionメッセージを送信可能 → 動的なsubscribe制御に使用

## 3. MessageQueue (ShmRingBuffer) ZMQフォールバック [VERIFIED]

EngineCore↔Worker間のSharedMemory通信で、メッセージがShmRingBufferの最大チャンクサイズ（デフォルト24MiB）を超えた場合にZMQ PUB/SUBへフォールバックする。

**参照**: `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py:590-594`

### オーバーフロー判定

```python
if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
    with self.acquire_write(timeout) as buf:
        buf[0] = 1  # overflow flag
    self.local_socket.send_multipart(all_buffers, copy=False)  # ZMQ XPUB
```

1. ShmRingBufferのメタデータブロックに `overflow=1` フラグを書き込み
2. 実データはXPUB→SUBソケット経由で送信
3. Reader側はメタデータのoverflowフラグを確認し、ZMQソケットから読み取る

### ソケット構成

- **Writer**: `XPUB` (bind、IPC)。ローカルリーダー向け + リモートリーダー向けの2つ
- **Local Reader**: `SUB` (connect、IPC)。ShmRingBufferとペア
- **Remote Reader**: `SUB` (connect、TCP)。ShmRingBufferなし、常にZMQ経由

### 接続確認

`wait_until_ready()`で全リーダーのサブスクリプション受信→`b"READY"`送信で双方向の接続を確認後に通信開始。

## 4. KV Cache Events [VERIFIED]

KVキャッシュの変更イベント（BlockStored/BlockRemoved/AllBlocksCleared）を外部サービスに配信する。

**参照**: `target/vllm/vllm/distributed/kv_events.py:270-400`

### ソケット構成

- **PUB** (bind、TCP `tcp://*:5557`): イベントストリームの配信
- **ROUTER** (bind): リプレイリクエストの受け付け（過去イベントの再送要求用）

### 特徴

- **HWM設定あり**: `set_hwm(100_000)` — PUBソケットにHWMが設定されている。サブスクライバが遅い場合、HWMを超えたメッセージはドロップされる
- **シーケンス番号**: 各イベントバッチにシーケンス番号を付与
- **リプレイ機能**: ROUTERソケットでリプレイリクエストを受け付け、バッファリングされた過去イベントを再送可能（deque、maxlen=10,000ステップ）
- **バックグラウンドスレッド**: パブリッシャーは専用スレッドで動作

## 5. KV Transfer コネクタ [VERIFIED]

ノード間KVキャッシュ転送の制御プレーンにZMQを使用。データプレーンは各コネクタ固有（RDMA、NCCL等）。

### 共通パターン: ROUTER/DEALER

全コネクタで共通して ROUTER（サーバー側、bind）/ DEALER（クライアント側、connect）パターンを使用。

| コネクタ | ZMQ用途 | ソケットタイプ |
|---------|---------|--------------|
| NIXL | メタデータハンドシェイク | ROUTER/REQ |
| P2P NCCL | 転送要求・応答 | ROUTER/DEALER |
| Mooncake | サイドチャネル通知 | ROUTER/DEALER |
| MoRIIO | メタデータ交換・通知 | ROUTER/DEALER |
| LMCache MP | LookupClient/Server通信 | ZMQ経由（LMCache内部） |

### NIXL特有: REQ/REP風ハンドシェイク

NIXLはROUTER/REQパターンを使用し、RDMAメモリ登録のためのメタデータ交換を行う。`RCVTIMEO=5000ms`でタイムアウトを設定。

**参照**: `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:615-618` (ROUTER側)
**参照**: `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:1062-1076` (REQ側)

## 信頼性分析

### ZMQの特性（前提知識）

ZMQは**メッセージ到達保証を提供しない**メッセージングライブラリ:
- TCPの上に構築されているが、接続のライフサイクル管理は自動（再接続含む）
- **PUB/SUB**: サブスクライバが遅い場合、HWMを超えたメッセージはサイレントにドロップされる
- **PUSH/PULL**: HWMに達するとブロック（またはEAGAIN）
- **ROUTER/DEALER**: HWMに達するとROUTER側はメッセージをドロップ（DEALERはブロック）
- IPC（同一ホスト内）はTCPより信頼性が高い（ネットワーク障害なし）

### 各通信パスの信頼性評価

#### 1. Frontend↔EngineCore（最重要パス）

**リスク**: 低

理由:
- **HWM=0（無制限）**: 送信側でのメッセージドロップは発生しない
- **同一ホスト内IPC**: ネットワーク障害のリスクなし（DP TCPモード除く）
- **プロセス死活監視**: `MPClientEngineMonitor`スレッドがEngineCoreプロセスのsentinelを監視。プロセス死亡→`engine_dead=True`→以降の操作は`EngineDeadError`
- **ENGINE_CORE_DEAD通知**: EngineCore異常終了時にFrontendへ通知。`linger=4000ms`で送信完了を待つ
- **validate_alive()**: 受信メッセージがENGINE_CORE_DEADか毎回チェック

**メッセージ喪失時の影響**:
- **リクエスト喪失（Frontend→EngineCore）**: リクエストが処理されず、クライアントがタイムアウトするまで待機。vLLM内部でのリトライ機構はない
- **レスポンス喪失（EngineCore→Frontend）**: リクエスト結果が返却されず、クライアントがタイムアウト。Schedulerのリクエストは残り続け、abort時に解放

**実際にメッセージ喪失が起きうるか**: IPCかつHWM=0の環境では、プロセスが正常動作している限りメッセージ喪失は極めて起きにくい。主なリスクはプロセスクラッシュ。

#### 2. DP Coordinator通信

**リスク**: 低〜中

理由:
- **XPUBの統計配信はベストエフォート**: 統計メッセージの一部が失われても負荷分散の精度が一時的に低下するだけ。定期的に再送されるため自己回復する
- **Wave開始指示の喪失**: エンジンが一時的にidle状態のまま留まる可能性がある。ただしフロントエンド側からの新リクエスト送信で再度waveが開始されるため、長時間のデッドロックにはならない
- **接続確認**: 全サブスクライバのsubscription受信を待ってから通信開始

**参照**: `target/vllm/vllm/v1/engine/coordinator.py:189-198` (サブスクリプション待ち)

#### 3. ShmRingBufferフォールバック

**リスク**: 低

理由:
- **XPUB/SUBだがHWM未設定（デフォルト1000）**: 理論上、非常に大きなメッセージが連続すると滞留可能
- **同一ホスト内IPC**: ネットワーク障害なし
- **頻度が低い**: 24MiB超のメッセージは稀（通常のSchedulerOutput、ModelRunnerOutputは小さい）
- **オーバーフロー自体がレアパス**: 通常はShmRingBuffer直接で完結

**メッセージ喪失時の影響**: Worker側でRPCレスポンスが受信できず、EngineCore側でハングする可能性

#### 4. KV Cache Events

**リスク**: 中（設計上許容）

理由:
- **PUBにHWM=100,000**: サブスクライバが遅い場合、メッセージがドロップされる
- **TCP経由**: ネットワーク障害のリスクあり
- **リプレイ機能で緩和**: シーケンス番号ギャップを検出し、ROUTERソケット経由で過去イベントを再取得可能
- **外部サービス向け**: vLLMのコア動作には影響しない。あくまでKVイベントの外部通知

**メッセージ喪失時の影響**: 外部キャッシュマネージャがイベントを見逃す。リプレイで回復可能（バッファ範囲内）

#### 5. KV Transfer コネクタ

**リスク**: 中

理由:
- **TCP経由（ノード間通信）**: ネットワーク障害のリスクあり
- **NIXL: RCVTIMEO設定あり**: 1000msまたは5000msのタイムアウト。`zmq.Again`例外をキャッチしてリトライロジックを実装
- **P2P NCCL**: Pollerでの待機、明示的なタイムアウトなし（ブロッキング）
- **Mooncake**: DEALER側にlinger=0設定。タイムアウト付きで転送

**メッセージ喪失時の影響**:
- ハンドシェイク失敗→コネクタ初期化失敗→ログエラー、該当リクエストは通常の計算パスにフォールバック
- 転送通知失敗→送信側がブロックをタイムアウト解放→受信側はprefillを再実行

### 信頼性設計のまとめ

vLLMのZMQ使用における信頼性戦略:

| 戦略 | 適用箇所 | 説明 |
|------|---------|------|
| **HWM=0（無制限バッファ）** | Frontend↔EngineCore | メッセージドロップを完全に防止 |
| **IPC優先** | 同一ホスト内通信 | ネットワーク障害を排除 |
| **プロセス死活監視** | Frontend→EngineCore | sentinelによる即座のクラッシュ検出 |
| **ENGINE_CORE_DEAD通知 + linger** | EngineCore→Frontend | 異常終了の明示的通知を保証 |
| **ハンドシェイク** | 起動時 | 通信確立の確認後に運用開始 |
| **リプレイ機能** | KV Events | メッセージドロップ後の回復手段 |
| **タイムアウト + リトライ** | KV Transfer | ネットワーク障害時のフォールバック |
| **ベストエフォート + 自己回復** | DP Coordinator | 統計は定期再送、waveは再トリガー |

**結論**: vLLMは**コア通信パスでは実質的にメッセージ喪失が起きない設計**（HWM=0 + IPC + プロセス監視）を採用し、**補助的な通信パスではベストエフォート + リカバリ機構**（リプレイ、タイムアウト、再トリガー）で対処している。ZMQの「到達保証なし」の弱点は、使用パターンの選択（IPC、HWM=0）とアプリケーション層の監視で効果的に緩和されている。

## ソケットタイプ使用一覧

| ソケットタイプ | 使用箇所 | 方向 | 特徴 |
|--------------|---------|------|------|
| `ROUTER` | Frontend input, NIXL server, P2P server, Mooncake server, MoRIIO server, KV Events replay, ハンドシェイク | bind（サーバー） | identityベースルーティング |
| `DEALER` | EngineCore input, NIXL client, P2P client, Mooncake client, MoRIIO client | connect（クライアント） | 透過的なidentity送信 |
| `PUSH` | EngineCore output, EngineCore→Coordinator | connect | 単方向、ブロック型 |
| `PULL` | Frontend output, Coordinator←EngineCore | bind | 単方向、フェアキューイング |
| `XPUB` | Coordinator→Frontend, Coordinator→EngineCore, MessageQueue writer | bind | サブスクリプション可視化 |
| `XSUB` | EngineCore←Coordinator, Frontend←Coordinator | connect | 明示的サブスクリプション |
| `SUB` | MessageQueue reader | connect | 自動サブスクリプション |
| `PUB` | KV Events | bind | ブロードキャスト、HWMドロップ |
| `PAIR` | Frontend内部（shutdown通知、first_req通知） | bind/connect | 排他的1:1ペア |
| `REQ` | NIXL ハンドシェイクclient | connect | 同期的リクエスト/レスポンス |

## 参照

| ファイル | 行 | 内容 |
|---------|-----|------|
| `target/vllm/vllm/v1/engine/core_client.py` | L510-514 | Frontend側ZMQソケット作成（ROUTER/PULL） |
| `target/vllm/vllm/v1/engine/core_client.py` | L539-549 | ROUTER初期メッセージ待ち（poll + タイムアウト） |
| `target/vllm/vllm/v1/engine/core_client.py` | L581-587 | MessageTracker管理（ゼロコピー参照保持） |
| `target/vllm/vllm/v1/engine/core_client.py` | L684-720 | SyncMPClientの出力処理スレッド（Poller + PAIR shutdown） |
| `target/vllm/vllm/v1/engine/core_client.py` | L877-901 | AsyncMPClientの出力処理タスク |
| `target/vllm/vllm/v1/engine/core_client.py` | L1080-1186 | DPClient統計購読（XSUB + PAIR first_req） |
| `target/vllm/vllm/v1/engine/core.py` | L870-920 | EngineCore側ハンドシェイク（DEALER→ROUTER） |
| `target/vllm/vllm/v1/engine/core.py` | L1186-1265 | EngineCore入力スレッド（DEALER + XSUB + Poller） |
| `target/vllm/vllm/v1/engine/core.py` | L1267-1335 | EngineCore出力スレッド（PUSH, tracker, linger=4000） |
| `target/vllm/vllm/v1/engine/coordinator.py` | L113-395 | DPCoordinator（XPUB×2 + PULL, Wave調整） |
| `target/vllm/vllm/v1/engine/utils.py` | L937-1091 | ハンドシェイクプロトコル（HELLO→metadata→READY） |
| `target/vllm/vllm/utils/network_utils.py` | L260-313 | make_zmq_socket（HWM=0, buf_size, IPv6対応） |
| `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py` | L280-403 | MessageQueue（XPUB/SUB, ShmRingBufferフォールバック） |
| `target/vllm/vllm/distributed/device_communicators/shm_broadcast.py` | L571-594 | enqueue（オーバーフロー判定→ZMQ送信） |
| `target/vllm/vllm/distributed/kv_events.py` | L270-400 | ZmqEventPublisher（PUB + ROUTER replay） |
| `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` | L615-618 | NIXL ROUTER（RCVTIMEO=1000ms） |
| `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py` | L124-130 | P2P NCCL ROUTER/DEALER |

## 関連ドキュメント

- [プロセスアーキテクチャ（TP=2構成）](process-architecture.md) — ShmRingBuffer、通信方式選択理由の詳細
- [Executorコンポーネント](../components/executor/summary.md) — MessageQueue、WorkerProc busy loop
- [EngineCoreClientコンポーネント](../components/engine-core-client/summary.md) — Frontend側の通信クライアント階層
- [KV Transferコンポーネント](../components/kv-transfer/summary.md) — KVConnectorBase_V1、各コネクタの概要
