# Phase 2d: EncoderCache・ECConnectorコンポーネント文書化

**日付**: 2026-02-14
**フェーズ**: Phase 2（コンポーネント別深堀り）
**トピック**: EncoderCache + ECConnector

## 概要

submoduleを最新mainブランチ（c027541ea）に更新後、EncoderCacheとECConnectorを独立したコンポーネントとして調査・文書化した。以前のinvestigation（`encoder-cache-persistence.md`, `ec-connector-github-discussions.md`）で得た知見をベースに、最新コードの構造を正確にマッピングした。

## 調査対象ファイル

| ファイル | 内容 |
|---|---|
| `vllm/v1/core/encoder_cache_manager.py` | EncoderCacheManager + EncoderDecoderCacheManager |
| `vllm/distributed/ec_transfer/ec_connector/base.py` | ECConnectorBase抽象基底クラス |
| `vllm/distributed/ec_transfer/ec_connector/factory.py` | ECConnectorFactory |
| `vllm/distributed/ec_transfer/ec_connector/example_connector.py` | ECExampleConnector参照実装 |
| `vllm/distributed/ec_transfer/ec_transfer_state.py` | シングルトン管理 |
| `vllm/distributed/ec_transfer/__init__.py` | 公開API |
| `vllm/v1/worker/ec_connector_model_runner_mixin.py` | GPUModelRunner統合Mixin |
| `vllm/config/ec_transfer.py` | ECTransferConfig |
| `vllm/v1/core/sched/scheduler.py` (部分) | ECConnector統合箇所 |
| `vllm/v1/worker/gpu_model_runner.py` (部分) | encoder_cache使用箇所 |
| `vllm/v1/engine/core.py` (部分) | is_ec_producer判定 |
| `vllm/v1/worker/gpu_worker.py` (部分) | Worker初期化 |
| `vllm/v1/outputs.py` (部分) | ECConnectorOutput |

## 主要な発見

### EncoderCacheManager

1. **FIFO遅延解放**: 参照カウント0のエントリはfreeableに移動するが即座には解放しない。新規割当時に空き不足の場合のみ古い順にEviction
2. **共有キャッシュ**: 同じmm_hashを持つエンコーダ出力は複数リクエスト間で共有。cached dictのvalueがrequest IDのsetで管理
3. **2層構造**: Scheduler側（論理管理）とWorker側（物理ストレージ dict[str, Tensor]）の明確な分離
4. **EncoderDecoderCacheManager**: Whisper等のEncoder-Decoderモデル用暫定実装。キャッシュ共有なし、毎回計算。1stepバッファで遅延解放

### ECConnector

1. **2ロール分離**: 同じクラスがScheduler側（has_cache_item, build_connector_meta）とWorker側（start_load_caches, save_caches）を担う。生成時にroleを渡す
2. **Mixin統合**: ECConnectorModelRunnerMixinがコンテキストマネージャでライフサイクル管理（bind→load→yield→get_finished→clear）
3. **Producer専用モード**: エンコーダ実行後デコーダスキップ、KVキャッシュ確保なし
4. **ECConnectorOutput未消費**: Worker→Schedulerフィードバックの`ec_connector_output`はModelRunnerOutputに含まれるがScheduler側で読まれていない
5. **request_finished未統合**: ECConnectorBase.request_finished()はSchedulerから呼ばれていない（KVConnectorのrequest_finishedは呼ばれている）

### Scheduler統合

- `_schedule_encoder_inputs()`内でECConnectorヒット時はcompute_budgetを消費しない（エンコーダ計算不要）
- ただしencoder_cache_manager.allocate()は実行（GPU側空きが必要）
- `build_connector_meta()`はSchedulerOutput構築時に呼ばれてSchedulerOutputに格納

## 成果物

- `docs/src/components/encoder-cache/summary.md` — [MEDIUM] [VERIFIED]
- `docs/src/components/ec-connector/summary.md` — [MEDIUM] [VERIFIED]

## 未解決の疑問

- ECConnectorOutput未消費の理由と今後の計画
- 既存の`encoder-cache-persistence.md`調査レポートとの重複部分の整理検討
