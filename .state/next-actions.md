# 次にやるべきこと

## 最優先

- [ ] Phase 1 セッション3: 下流パス追跡（Executor → Worker → GPUModelRunner → OutputProcessor）
  - Executor/Worker の委譲パターン把握
  - GPUModelRunner.execute_model() の概要（詳細はPhase 2）
  - OutputProcessor → RequestOutput 変換とDetokenize
  - data-flow.md 完成（下流パスセクション、Prefill vs Decode、コンポーネント優先度確定）
  - 成果物: data-flow.md 完成版 + コンポーネントsummary × 3（executor, gpu-model-runner, output-processor）

## 次点

- [ ] Phase 2 開始: KVCacheManager 深堀り（ユーザー関心1位）
  - BlockPool の詳細（Evictionアルゴリズム、FreeKVCacheBlockQueue）
  - プレフィックスキャッシュのハッシュ計算詳細
  - allocate_slots() の全分岐パス
  - KVCacheCoordinator / SingleTypeKVCacheManager の内部

## いつかやる

- [ ] KV Transfer / LMCache のコンポーネントsummary.md作成（Phase 2で詳細調査）
- [ ] マルチモーダル（`vllm/multimodal/`）のコンポーネントsummary.md作成
- [ ] プラグインシステム（`vllm/plugins/`）の仕組み把握
- [ ] C++/CUDAカーネル（`csrc/`）のAPI一覧整理
- [ ] 分散推論（Tensor/Pipeline並列）の仕組み把握
- [ ] Speculative Decodingの実装詳細
- [ ] batch_queue パイプライン並列化の実践的な動作（max_concurrent_batches > 1）
- [ ] block_size の設定方法とパフォーマンスへの影響
