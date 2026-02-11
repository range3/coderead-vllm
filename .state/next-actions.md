# 次にやるべきこと

## 最優先

- [ ] Phase 2 続行: GPUModelRunner 深堀り（優先度A）
  - Attentionメタデータ構築（KVCacheManagerのブロック情報をどう参照するか）
  - KVConnectorModelRunnerMixin（KV Transfer、ユーザー関心2位）
  - マルチモーダル入力処理（ユーザー関心3位）
  - _prepare_inputs() と _update_states() の詳細
  - 成果物: gpu-model-runner/summary.md を [SHALLOW]→[MEDIUM] 以上に昇格

## 次点

- [ ] Phase 2: Scheduler 深堀り（優先度A）
  - _update_request_with_output() の完了判定詳細
  - Speculative Decoding のリジェクション処理
  - チャンクプリフィルのトークン予算分割

- [ ] Phase 2: KV Transfer / LMCache 調査（ユーザー関心2位）
  - KVConnector 抽象基底クラスの仕組み
  - LMCacheConnector の実装
  - KV Cache Events との連携
  - コンポーネント summary.md 作成

## いつかやる

- [ ] マルチモーダル（`vllm/multimodal/`）のコンポーネントsummary.md作成
- [ ] プラグインシステム（`vllm/plugins/`）の仕組み把握
- [ ] C++/CUDAカーネル（`csrc/`）のAPI一覧整理
- [ ] 分散推論（Tensor/Pipeline並列）の仕組み把握
- [ ] Speculative Decodingの実装詳細
- [ ] batch_queue パイプライン並列化の実践的な動作（max_concurrent_batches > 1）

## 完了

- [x] Phase 2: KVCacheManager 深堀り（2026-02-11）
  - BlockPool（FreeKVCacheBlockQueue、BlockHashToBlockMap、Eviction、null_block）
  - プレフィックスキャッシュ（ハッシュチェーン、Extra Keys、4種Lookupアルゴリズム、Hybrid fixed-point）
  - アテンションタイプ別Manager 7種（Full/SW/Chunked/Mamba/Cross/Sink）
  - summary.md [MEDIUM]→[DEEP] 昇格
