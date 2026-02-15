# Summary

[はじめに](README.md)


---

# アーキテクチャ

- [テキスト推論データフロー](architecture/data-flow.md)
- [アーキテクチャ概要](architecture/overview.md)


---

# コンポーネント

- [ECConnector（Encoder Cache Connector）](components/ec-connector/summary.md)
- [EncoderCache（エンコーダキャッシュ）](components/encoder-cache/summary.md)
- [EngineCore サマリー](components/engine-core/summary.md)
- [EngineCoreClient サマリー](components/engine-core-client/summary.md)
- [エントリポイント (AsyncLLM / LLM) サマリー](components/entrypoint/summary.md)
- [Executor](components/executor/summary.md)
- [GPUModelRunner](components/gpu-model-runner/summary.md)
  - [InputBatch: 永続バッチと状態管理](components/gpu-model-runner/input-batch.md)
  - [KVCache-GPU Interface: ブロックテーブルとスロットマッピング](components/gpu-model-runner/kv-cache-interface.md)
- [InputProcessor サマリー](components/input-processor/summary.md)
- [KVCacheManager サマリー](components/kv-cache-manager/summary.md)
  - [アテンションタイプ別 Manager 詳細](components/kv-cache-manager/attention-type-managers.md)
  - [BlockPool 詳細](components/kv-cache-manager/block-pool.md)
  - [プレフィックスキャッシュ詳細](components/kv-cache-manager/prefix-cache.md)
- [KV Transfer [MEDIUM] [VERIFIED]](components/kv-transfer/summary.md)
- [マルチモーダル処理パイプライン サマリー](components/multimodal/summary.md)
  - [Gemma3 ビジョンエンコーダと画像処理 [MEDIUM] [VERIFIED]](components/multimodal/gemma3-vision.md)
  - [バックエンド マルチモーダル処理パス [MEDIUM] [VERIFIED]](components/multimodal/mm-engine-gpu.md)
  - [フロントエンド マルチモーダル処理パス [MEDIUM] [VERIFIED]](components/multimodal/mm-processing.md)
- [OutputProcessor](components/output-processor/summary.md)
- [Scheduler サマリー](components/scheduler/summary.md)


---

# 調査報告

- [CacheBlend GitHub 議論調査](investigations/cacheblend-github-discussions.md)
- [CacheBlend 実装調査報告 [MEDIUM] [VERIFIED]](investigations/cacheblend-implementation.md)
- [ECConnector GitHub 議論調査レポート](investigations/ec-connector-github-discussions.md)
- [EncoderCache 永続化と階層キャッシュ: 調査報告](investigations/encoder-cache-persistence.md)
- [Gemma3 ビジョンパイプライン: キャッシュ機構 [MEDIUM] [VERIFIED]](investigations/gemma3-vision-caches.md)
- [Gemma3 27B ビジョンパイプライン: 形状フローと数値まとめ](investigations/gemma3-vision-pipeline.md)
- [LMCache 統合調査報告 [MEDIUM] [VERIFIED]](investigations/lmcache-integration.md)
- [プロセスアーキテクチャ（TP=2構成）](investigations/process-architecture.md)

---

# 付録

- [用語集](glossary.md)
- [ファイル索引](appendix/file-index.md)

