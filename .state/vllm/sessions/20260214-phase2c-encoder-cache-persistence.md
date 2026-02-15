# Phase 2c: EncoderCache 永続化・階層キャッシュ化調査

**日付**: 2026-02-14
**目的**: EncoderCache の FIFO→LRU 変更と、ストレージバックエンドによる階層キャッシュの実現可能性調査

## 主要な発見

### 1. ECConnector 既存インフラの発見

vLLM には KV Transfer とは完全に独立した `ECConnector`（Encoder Cache Connector）枠組みが既に存在。

- **ディレクトリ**: `target/vllm/vllm/distributed/ec_transfer/`
- **基底クラス**: `ECConnectorBase`（5 abstract メソッド）
- **参照実装**: `ECExampleConnector`（safetensors でディスク保存、199 行）
- **ファクトリ**: `ECConnectorFactory`（プラグイン登録、動的ロード対応）
- **設定**: `ECTransferConfig`（`--ec-connector`, `--ec-role`）
- **統合済み**: GPUModelRunner（`ec_connector_model_runner_mixin.py`）+ Scheduler（`scheduler.py:1197-1203`）

### 2. KV Transfer は不適合

- LMCache は `kv_shape = (num_layer, 2, chunk_size, num_kv_heads, head_size)` がハードコード
- エンコーダ出力テンソル `(N×256, 5376)` とは形状・粒度が合わない
- ECConnector はテンソル形状に一切依存しない

### 3. FIFO→LRU は 1 ファイル変更で実現可能

- `encoder_cache_manager.py` の `check_and_update_cache()` と `free_encoder_input()` の 2 メソッド修正
- Scheduler 側・GPUModelRunner 側の変更は不要

### 4. 階層キャッシュは ECConnector + LRU 化で実現可能

- L1: GPU dict（LRU）+ L2: カスタム ECConnector（Redis/ディスク等）
- フロー: L1 HIT → スキップ / L1 MISS, L2 HIT → ロード / L1/L2 MISS → エンコーダ実行 + L2 保存

## 調査対象ファイル

| ファイル | 内容 |
|---|---|
| `target/vllm/vllm/v1/core/encoder_cache_manager.py` | EncoderCacheManager 全体 |
| `target/vllm/vllm/distributed/ec_transfer/ec_connector/base.py` | ECConnectorBase |
| `target/vllm/vllm/distributed/ec_transfer/ec_connector/example_connector.py` | ECExampleConnector |
| `target/vllm/vllm/distributed/ec_transfer/ec_connector/factory.py` | ECConnectorFactory |
| `target/vllm/vllm/config/ec_transfer.py` | ECTransferConfig |
| `target/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py` | Mixin |
| `target/vllm/vllm/v1/core/sched/scheduler.py:1197-1203` | ECConnector 統合 |
| `target/vllm/vllm/v1/worker/gpu_model_runner.py:2442-2445` | save_caches 呼び出し |
| `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/base.py` | KVConnectorBase_V1（比較用） |
| `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` | LMCache 形状ハードコード確認 |

## 成果物

- `docs/src/investigations/encoder-cache-persistence.md` — 調査報告（新規作成）
- `docs/src/investigations/gemma3-vision-caches.md` — リンク追加
