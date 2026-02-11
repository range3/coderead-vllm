# Summary

[はじめに](README.md)


---

# アーキテクチャ

- [テキスト推論データフロー](architecture/data-flow.md)
- [アーキテクチャ概要](architecture/overview.md)


---

# コンポーネント

- [EngineCore サマリー](components/engine-core/summary.md)
- [EngineCoreClient サマリー](components/engine-core-client/summary.md)
- [エントリポイント (AsyncLLM / LLM) サマリー](components/entrypoint/summary.md)
- [Executor](components/executor/summary.md)
- [GPUModelRunner](components/gpu-model-runner/summary.md)
- [InputProcessor サマリー](components/input-processor/summary.md)
- [KVCacheManager サマリー](components/kv-cache-manager/summary.md)
  - [アテンションタイプ別 Manager 詳細](components/kv-cache-manager/attention-type-managers.md)
  - [BlockPool 詳細](components/kv-cache-manager/block-pool.md)
  - [プレフィックスキャッシュ詳細](components/kv-cache-manager/prefix-cache.md)
- [マルチモーダル処理パイプライン サマリー](components/multimodal/summary.md)
  - [Gemma3 ビジョンエンコーダと画像処理 [MEDIUM] [VERIFIED]](components/multimodal/gemma3-vision.md)
  - [バックエンド マルチモーダル処理パス [MEDIUM] [VERIFIED]](components/multimodal/mm-engine-gpu.md)
  - [フロントエンド マルチモーダル処理パス [MEDIUM] [VERIFIED]](components/multimodal/mm-processing.md)
- [OutputProcessor](components/output-processor/summary.md)
- [Scheduler サマリー](components/scheduler/summary.md)


---

# 調査報告

- [Gemma3 27B ビジョンパイプライン: 形状フローと数値まとめ](investigations/gemma3-vision-pipeline.md)

---

# 付録

- [用語集](glossary.md)
- [ファイル索引](appendix/file-index.md)

