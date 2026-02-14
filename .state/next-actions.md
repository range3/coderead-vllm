# 次にやるべきこと

## 最優先

- [ ] Phase 2: マルチモーダル DEEP化（ユーザー関心3位、MEDIUM済み）
  - ProcessorCache shm モードの詳細（SharedMemory Ring Buffer）
  - BaseMultiModalProcessor.apply() の内部フロー詳細
  - MM × プレフィックスキャッシュの相互作用
  - Gemma3以外のMMモデルとの差分パターン

## 次点

- [ ] Phase 2: Scheduler 深堀り（優先度A）
  - Speculative Decoding のリジェクション処理
  - チャンクプリフィルのトークン予算分割

- [ ] Phase 2: GPUModelRunner DEEP化（MEDIUM済み）
  - async_scheduling の _update_states_after_model_execute() 詳細
  - Speculative Decoding 提案メソッド（propose_draft_token_ids）
  - ForwardContextでのreshape_and_cache()消費詳細
  - Attentionバックエンド初期化（initialize_attn_backend / AttentionGroup）

- [ ] Phase 2: KV Transfer DEEP化（MEDIUM済み）
  - NixlConnector の詳細（RDMA、pre-register、handshake）
  - OffloadingConnector の詳細（CPU/ディスクオフロード）
  - MultiConnector の複合パターン
  - LMCache CacheBlend（blending）の動作
  - cross-layer blocks の実際の性能影響

## いつかやる

- [ ] プラグインシステム（`vllm/plugins/`）の仕組み把握
- [ ] C++/CUDAカーネル（`csrc/`）のAPI一覧整理
- [x] 分散推論（Tensor/Pipeline並列）の仕組み把握 — プロセスアーキテクチャ調査で部分完了（TP=2構成、`docs/src/investigations/process-architecture.md`）。PP詳細は未調査
- [ ] Speculative Decodingの実装詳細
- [ ] batch_queue パイプライン並列化の実践的な動作（max_concurrent_batches > 1）
- [ ] EncoderCache FIFO→LRU 変更の実装（設計完了、`encoder-cache-persistence.md` 参照）
- [ ] カスタム ECConnector の実装（Redis/S3等のバックエンド）
- [ ] Mooncake ECConnector の進展をウォッチ（#33714 コメント、fake0fan fork）
- [ ] エンコーダキャッシュ事前割り当て方式の動向追跡（dict→固定バッファ移行の可能性）

## 完了

- [x] Phase 2g: KV Transfer / LMCache 調査（2026-02-15）
  - KVConnectorBase_V1: 7 abstract メソッド、2ロール分離（Scheduler/Worker）
  - KVConnectorFactory: 10個の登録済みコネクタ、遅延ロード
  - Scheduler統合: WAITING_FOR_REMOTE_KVS状態、外部キャッシュ問い合わせ、遅延解放
  - Worker/GPUModelRunner統合: KVConnectorModelRunnerMixin、レイヤー別パイプライニング
  - KV Cache Events: BlockStored/Removed/AllBlocksCleared、ZMQ配信
  - LMCache: チャンク単位保存（256トークン）、3層ストレージ、15+コネクタ
  - vLLMアダプタ: native/latest分岐、RequestTracker/ReqMeta
  - summary.md [SHALLOW]→[MEDIUM] 昇格 + investigations作成

- [x] Phase 2f: GPUModelRunner 深堀り（2026-02-15）
  - KVCache-GPU Interface: ブロックID取込→BlockTable→slot_mapping→DMA→AttentionMetadata
  - InputBatch永続バッチ: CachedRequestState/InputBatch/MultiGroupBlockTable/CpuGpuBuffer/condense
  - CUDAGraph統合: 3モード（FULL/PIECEWISE/NONE）、CudagraphDispatcher、パディング
  - summary.md [SHALLOW]→[MEDIUM] 昇格 + 2サブドキュメント作成

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

- [x] Phase 2c: EncoderCache永続化・階層キャッシュ化調査（2026-02-14）
- [x] Phase 2d: EncoderCache・ECConnectorコンポーネント文書化（2026-02-14）
