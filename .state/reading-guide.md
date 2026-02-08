# vLLM 読解ガイド

> **確信度**: [INFERRED]
> **最終更新**: 2026-02-09

## コードベース構造ルール

### アーキテクチャ世代

- **フォーカス**: `vllm/v1/` が現行アーキテクチャの本体
- `vllm/engine/llm_engine.py` は `from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine` の1行エイリアス [VERIFIED]
- `vllm/engine/async_llm_engine.py` も同様に `vllm.v1.engine.async_llm.AsyncLLM` へのエイリアス [VERIFIED]
- **ただし** `vllm/model_executor/`、`vllm/distributed/`、`vllm/multimodal/`、`vllm/lora/` 等はv1から直接利用されるため調査対象に含む

### 代表実装パターン

- **リファレンス**: `vllm/model_executor/models/gemma3.py` を基準とする（マルチモーダル対応モデル）
- **差分記録**: 他240+モデルは Gemma3 との差分のみ記録
- **理由**: Gemma3は画像入力を含むマルチモーダル対応であり、ユーザーの関心領域に合致

### プラットフォーム/バックエンド

- **主要**: CUDA を深堀り（`gpu_worker.py`, `gpu_model_runner.py`, `csrc/`）
- **差分記録**: ROCm, CPU, TPU, XPU, Gaudi は CUDA からの差分のみ記録

### スキップ対象

- `benchmarks/` — 性能調査時のみ参照
- `tests/` — テストパターン確認時のみ参照（実装理解には不要）
- `docs/` (target内) — 公式ドキュメントとして参照するがコードリーディング対象外
- `tools/` — 開発用ユーティリティ
- `docker/`, `.github/`, `.buildkite/` — CI/CD基盤
- `vllm/entrypoints/openai/` のHTTPハンドラ詳細 — FastAPIルーティングの詳細は深追い不要（API仕様はドキュメント参照）

### 推奨読み順

```
AsyncLLM (vllm/v1/engine/async_llm.py)
  → EngineCore (vllm/v1/engine/core.py)
    → Scheduler (vllm/v1/core/sched/scheduler.py)
    → KVCacheManager (vllm/v1/core/kv_cache_manager.py)
    → Executor (vllm/v1/executor/)
      → Worker (vllm/v1/worker/gpu_worker.py)
        → GPUModelRunner (vllm/v1/worker/gpu_model_runner.py)
          → Models (vllm/model_executor/models/gemma3.py)
```

### C++/CUDAカーネルの読み方

- `csrc/` 配下のCUDA実装はパフォーマンスクリティカルだが、制御フロー把握にはPythonバインディングのAPI理解で十分
- まず `vllm/_custom_ops.py` 等のバインディング層でAPIを確認し、必要な場合のみCUDAコードに入る
- FlashAttention、FlashInfer等の外部カーネルライブラリは呼び出しインターフェースのみ把握

## ユーザー優先度

### 関心領域（優先度順）

1. **メモリ管理/KVキャッシュ** — PagedAttention、KVCacheManager、BlockPool、ブロック管理の仕組み
2. **KV Transfer/LMCache連携** — Disaggregated Prefill、KVコネクタ抽象、LMCache統合の実装
3. **マルチモーダル** — 画像入力推論、mm_cache（マルチモーダルキャッシュ）、エンコーダ統合
4. **API/エントリポイント** — AsyncLLM、エンジン層、OpenAI互換API
5. **スケジューラ/推論パイプライン** — Continuous Batching、Scheduler実装、推論ループ全体

### 関心が低い領域

- テスト基盤、CI/CD設定
- ドキュメント生成
- ベンチマークツール
- 個別モデル実装の詳細（Gemma3以外）

### 調査の背景・動機

LLM推論サービングの仕組みを深く学ぶための技術理解・学習が目的。将来的に独自プラグインの作成も考慮している。特にKVキャッシュ関連（KV Transfer、LMCache連携）とマルチモーダル推論（画像入力、mm_cache）に強い関心がある。
