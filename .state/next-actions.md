# 次にやるべきこと

## 最優先

- [ ] Phase 2 続行: GPUModelRunner 深堀り（優先度A）
  - Attentionメタデータ構築（KVCacheManagerのブロック情報をどう参照するか）
  - CUDAGraph統合
  - _prepare_inputs() と _update_states() の詳細
  - 成果物: gpu-model-runner/summary.md を [SHALLOW]→[MEDIUM] 以上に昇格

## 次点

- [ ] Phase 2: KV Transfer / LMCache 調査（ユーザー関心2位）
  - KVConnector 抽象基底クラスの仕組み
  - LMCacheConnector の実装
  - KVConnectorModelRunnerMixin の動作
  - KV Cache Events との連携
  - コンポーネント summary.md 作成

- [ ] Phase 2: マルチモーダル DEEP化（ユーザー関心3位、MEDIUM済み）
  - ProcessorCache shm モードの詳細（SharedMemory Ring Buffer）
  - BaseMultiModalProcessor.apply() の内部フロー詳細
  - MM × プレフィックスキャッシュの相互作用
  - Gemma3以外のMMモデルとの差分パターン

- [ ] Phase 2: Scheduler 深堀り（優先度A）
  - Speculative Decoding のリジェクション処理
  - チャンクプリフィルのトークン予算分割

## いつかやる

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

- [x] Phase 2: マルチモーダル画像推論パス（2026-02-11）
  - フロントエンド: チャットテンプレート、プレースホルダー、MMHasher(blake3)、ProcessorCache 4種
  - バックエンド: EncoderCacheManager(RefCount+FIFO)、Schedulerエンコーダ予算、GPUModelRunnerエンコーダ実行
  - Gemma3: SiglipVisionModel、Gemma3MultiModalProjector、Pan-and-Scan、masked_scatter_マージ
  - summary.md + 3サブドキュメント作成、[SHALLOW]→[MEDIUM] 昇格
