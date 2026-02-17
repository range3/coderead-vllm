# Phase 3a: ZMQ通信パターン横断調査

**日付**: 2026-02-18
**Phase**: 3（横断的機能理解）
**テーマ**: vLLM全体のZMQ使用パターンと信頼性分析

## 実施内容

### 調査範囲
- vLLM v1内の全ZMQ使用ファイル（16ファイル）を特定
- 5カテゴリに分類して通信パターンを体系的に分析
- 各通信パスの信頼性とメッセージ喪失時の影響を評価

### 発見した5カテゴリ

1. **Frontend↔EngineCore通信**（5ファイル）: ROUTER/DEALER + PUSH/PULL。HWM=0で実質喪失なし
2. **DP Coordinator**（1ファイル）: XPUB/XSUB + PULL。Wave調整+統計配信
3. **MessageQueue ShmRingBufferフォールバック**（1ファイル）: XPUB/SUB。24MiB超メッセージのみ
4. **KV Cache Events**（1ファイル）: PUB + ROUTER。HWM=100K、リプレイ機能あり
5. **KV Transfer コネクタ**（8ファイル）: ROUTER/DEALER, REQ。ノード間制御プレーン

### 10種ソケットタイプ使用確認
ROUTER, DEALER, PUSH, PULL, XPUB, XSUB, SUB, PUB, PAIR, REQ

### 信頼性分析の結論
- コア通信（Frontend↔EngineCore）: HWM=0 + IPC + プロセス監視 → 実質メッセージ喪失なし
- 補助通信: ベストエフォート + リカバリ機構（リプレイ、タイムアウト、再トリガー）

## 成果物
- `docs/src/vllm/investigations/zmq-communication-patterns.md` — 調査報告書

## コンテキスト消費
- ソースコード読み: core_client.py, core.py, coordinator.py, utils.py, network_utils.py, shm_broadcast.py, kv_events.py, nixl_connector.py, p2p_nccl_engine.py, mooncake_connector.py, moriio_*.py, lmcache_mp_connector.py, multi_process_adapter.py
