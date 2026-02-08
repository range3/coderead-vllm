# 次にやるべきこと

## 最優先

- [ ] Phase 1 セッション2: コアループ追跡（EngineCore → Scheduler → KVCacheManager → Executor）
  - EngineCore.step() の schedule → execute → update サイクルを詳細追跡
  - Scheduler.schedule() のトークン予算割当ロジック概要把握
  - KVCacheManager.allocate_slots() のインターフェース把握
  - SchedulerOutput の全フィールドを整理
  - 成果物: data-flow.md コアループセクション + コンポーネントsummary × 3（engine-core, scheduler, kv-cache-manager）

## 次点

- [ ] Phase 1 セッション3: 下流パス追跡（Executor → Worker → GPUModelRunner → OutputProcessor）
  - Executor/Worker の委譲パターン把握
  - GPUModelRunner.execute_model() の概要（詳細はPhase 2）
  - OutputProcessor → RequestOutput 変換とDetokenize
  - data-flow.md 完成、コンポーネント優先度確定
  - 成果物: data-flow.md 完成版 + コンポーネントsummary × 3（executor, gpu-model-runner, output-processor）

## いつかやる

- [ ] KV Transfer / LMCache のコンポーネントsummary.md作成（Phase 2で詳細調査）
- [ ] マルチモーダル（`vllm/multimodal/`）のコンポーネントsummary.md作成
- [ ] プラグインシステム（`vllm/plugins/`）の仕組み把握
- [ ] C++/CUDAカーネル（`csrc/`）のAPI一覧整理
- [ ] 分散推論（Tensor/Pipeline並列）の仕組み把握
- [ ] Speculative Decodingの実装詳細
