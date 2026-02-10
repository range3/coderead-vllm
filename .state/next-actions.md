# 次にやるべきこと

## 最優先

- [ ] Phase 2 開始: KVCacheManager 深堀り（ユーザー関心1位）
  - BlockPool の詳細（Evictionアルゴリズム、FreeKVCacheBlockQueue）
  - プレフィックスキャッシュのハッシュ計算詳細
  - allocate_slots() の全分岐パス
  - KVCacheCoordinator / SingleTypeKVCacheManager の内部
  - 成果物: kv-cache-manager/summary.md を [MEDIUM]→[DEEP] に昇格

## 次点

- [ ] Phase 2: GPUModelRunner 深堀り（優先度A）
  - Attentionメタデータ構築（KVキャッシュとの連携点）
  - KVConnectorModelRunnerMixin（KV Transfer、ユーザー関心2位）
  - マルチモーダル入力処理（ユーザー関心3位）
  - _prepare_inputs() と _update_states() の詳細

- [ ] Phase 2: Scheduler 深堀り（優先度A）
  - _update_request_with_output() の完了判定詳細
  - Speculative Decoding のリジェクション処理
  - チャンクプリフィルのトークン予算分割

## いつかやる

- [ ] KV Transfer / LMCache のコンポーネントsummary.md作成（Phase 2で詳細調査）
- [ ] マルチモーダル（`vllm/multimodal/`）のコンポーネントsummary.md作成
- [ ] プラグインシステム（`vllm/plugins/`）の仕組み把握
- [ ] C++/CUDAカーネル（`csrc/`）のAPI一覧整理
- [ ] 分散推論（Tensor/Pipeline並列）の仕組み把握
- [ ] Speculative Decodingの実装詳細
- [ ] batch_queue パイプライン並列化の実践的な動作（max_concurrent_batches > 1）
- [ ] block_size の設定方法とパフォーマンスへの影響
