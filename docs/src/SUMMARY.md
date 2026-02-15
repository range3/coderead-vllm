# Summary

[はじめに](README.md)


---

# vLLM

- [vLLM](vllm/README.md)

# vLLM: アーキテクチャ

- [テキスト推論データフロー](vllm/architecture/data-flow.md)
- [アーキテクチャ概要](vllm/architecture/overview.md)

# vLLM: コンポーネント

- [ECConnector（Encoder Cache Connector）](vllm/components/ec-connector/summary.md)
- [EncoderCache（エンコーダキャッシュ）](vllm/components/encoder-cache/summary.md)
- [EngineCore サマリー](vllm/components/engine-core/summary.md)
- [EngineCoreClient サマリー](vllm/components/engine-core-client/summary.md)
- [エントリポイント (AsyncLLM / LLM) サマリー](vllm/components/entrypoint/summary.md)
- [Executor](vllm/components/executor/summary.md)
- [GPUModelRunner](vllm/components/gpu-model-runner/summary.md)
  - [InputBatch: 永続バッチと状態管理](vllm/components/gpu-model-runner/input-batch.md)
  - [KVCache-GPU Interface: ブロックテーブルとスロットマッピング](vllm/components/gpu-model-runner/kv-cache-interface.md)
- [InputProcessor サマリー](vllm/components/input-processor/summary.md)
- [KVCacheManager サマリー](vllm/components/kv-cache-manager/summary.md)
  - [アテンションタイプ別 Manager 詳細](vllm/components/kv-cache-manager/attention-type-managers.md)
  - [BlockPool 詳細](vllm/components/kv-cache-manager/block-pool.md)
  - [プレフィックスキャッシュ詳細](vllm/components/kv-cache-manager/prefix-cache.md)
- [KV Transfer [MEDIUM] [VERIFIED]](vllm/components/kv-transfer/summary.md)
- [マルチモーダル処理パイプライン サマリー](vllm/components/multimodal/summary.md)
  - [Gemma3 ビジョンエンコーダと画像処理 [MEDIUM] [VERIFIED]](vllm/components/multimodal/gemma3-vision.md)
  - [バックエンド マルチモーダル処理パス [MEDIUM] [VERIFIED]](vllm/components/multimodal/mm-engine-gpu.md)
  - [フロントエンド マルチモーダル処理パス [MEDIUM] [VERIFIED]](vllm/components/multimodal/mm-processing.md)
- [OutputProcessor](vllm/components/output-processor/summary.md)
- [Scheduler サマリー](vllm/components/scheduler/summary.md)

# vLLM: 調査報告

- [CacheBlend GitHub 議論調査](vllm/investigations/cacheblend-github-discussions.md)
- [CacheBlend 実装調査報告 [MEDIUM] [VERIFIED]](vllm/investigations/cacheblend-implementation.md)
- [ECConnector GitHub 議論調査レポート](vllm/investigations/ec-connector-github-discussions.md)
- [EncoderCache 永続化と階層キャッシュ: 調査報告](vllm/investigations/encoder-cache-persistence.md)
- [Gemma3 ビジョンパイプライン: キャッシュ機構 [MEDIUM] [VERIFIED]](vllm/investigations/gemma3-vision-caches.md)
- [Gemma3 27B ビジョンパイプライン: 形状フローと数値まとめ](vllm/investigations/gemma3-vision-pipeline.md)
- [LMCache 統合調査報告 [MEDIUM] [VERIFIED]](vllm/investigations/lmcache-integration.md)
- [プロセスアーキテクチャ（TP=2構成）](vllm/investigations/process-architecture.md)

# vLLM: 付録

- [用語集](vllm/glossary.md)
- [ファイル索引](vllm/appendix/file-index.md)

---

# LMCache

- [LMCache](lmcache/README.md)

# LMCache: アーキテクチャ

- [データフロー](lmcache/architecture/data-flow.md)
- [LMCache アーキテクチャ概要](lmcache/architecture/overview.md)

# LMCache: コンポーネント

- [LMCacheEngine](lmcache/components/cache-engine/summary.md)
- [GPUConnector](lmcache/components/gpu-connector/summary.md)
- [StorageManager + LocalCPUBackend](lmcache/components/storage-manager/summary.md)
- [TokenDatabase（ChunkedTokenDatabase）](lmcache/components/token-database/summary.md)
- [vLLM統合（LMCacheConnector）](lmcache/components/vllm-integration/summary.md)

# LMCache: 付録

- [用語集](lmcache/glossary.md)
