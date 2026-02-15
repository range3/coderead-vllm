# Phase 2b: マルチモーダル画像推論パス

**日付**: 2026-02-11
**フェーズ**: Phase 2b
**テーマ**: マルチモーダル（画像入力推論、mm_cache、ビジョンエンコーダ）

## 概要

Gemma3モデルでの画像を含む推論パスをエンドツーエンドで追跡。フロントエンド（P0）でのチャットテンプレート適用・HF Processor実行・プロセッサキャッシュ、バックエンド（P1）でのEncoderCacheManager・Schedulerエンコーダ予算・GPUModelRunnerエンコーダ実行・埋め込みマージ、Gemma3固有のSiglipVisionModel・MultiModalProjectorを調査した。

## 成果物

### 新規作成
- `docs/src/components/multimodal/summary.md` — マルチモーダル処理パイプライン全体像 [MEDIUM]
- `docs/src/components/multimodal/mm-processing.md` — フロントエンド側MM処理 [MEDIUM]
- `docs/src/components/multimodal/mm-engine-gpu.md` — バックエンド側MM処理 [MEDIUM]
- `docs/src/components/multimodal/gemma3-vision.md` — Gemma3固有ビジョン処理 [MEDIUM]

### 更新
- `docs/src/architecture/data-flow.md` — MM推論パスの差分セクション追加
- `docs/src/components/input-processor/summary.md` — MM関連処理追加
- `docs/src/components/gpu-model-runner/summary.md` — MM関連処理追加
- `docs/src/glossary.md` — 8用語追加（MultiModalFeatureSpec等）

## 主要な発見

### キャッシュの3層構造
1. **ProcessorCache（P0）**: HF Processor処理結果のキャッシュ。4種類（processor_only/lru/shm/none）
2. **EncoderCacheManager（P1 Scheduler）**: エンコーダ出力の論理管理。リファレンスカウント + FIFO遅延Eviction
3. **encoder_cache（P1 GPU）**: GPU上のテンソルキャッシュ。`dict[str, Tensor]`

### P0-P1キャッシュの整合性設計
- P0とP1のキャッシュEviction順序を同期
- `is_cached()` はP0のみ参照（IPC不要）
- `get_and_update()` をP0→P1の順で呼ぶことで整合性維持

### Gemma3ビジョンパイプライン
- SiglipVisionModel: Conv2d(patch) → position_embed → N層Transformer（双方向Attention）
- Gemma3MultiModalProjector: reshape → AvgPool2d → GemmaRMSNorm → Linear
- `masked_scatter_` でtext embedding + vision embeddingをin-placeマージ

### チャットテンプレートとプレースホルダー
- `<start_of_image>` → `full_image_sequence`（image_token × 256 + BOI/EOI）
- Pan-and-Scan有効時: original + crops で (num_crops+1) × 256 トークン
- 改行トークンの特殊結合処理（\n + \n\n → \n\n\n）

## 未解決の疑問
- ProcessorCacheのshmモードのSingleWriterShmRingBuffer詳細
- MM × プレフィックスキャッシュの相互作用（同じ画像でKVキャッシュヒットする条件）
- 複数画像入力時のエンコーダバッチ処理のパフォーマンス特性

## 次回に向けて
- GPUModelRunner深堀り（Attentionメタデータ、CUDAGraph）が最優先
- KV Transfer / LMCache調査がユーザー関心2位
- マルチモーダルDEEP化（ProcessorCache shm、BaseMultiModalProcessor.apply内部）
