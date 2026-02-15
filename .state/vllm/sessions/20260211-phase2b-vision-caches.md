# セッション記録: Phase 2b++ Gemma3ビジョンキャッシュ機構調査

**日付**: 2026-02-11
**フェーズ**: Phase 2b++（マルチモーダル追加調査）
**テーマ**: Gemma3ビジョンパイプラインの3層キャッシュ機構

## 目的

gemma3-vision-pipeline.md で追跡したパイプライン（Step 1〜8）上にある各キャッシュについて、ハッシュ入力・保存値・スキップされる処理・CPU/GPUを特定する。

## 調査結果

### 特定した3つのキャッシュ

1. **ProcessorCache**（CPU, P0）
   - キー: blake3(model_id, PILイメージピクセル, processor_kwargs, tokenizer_kwargs)
   - 保存: pixel_values + num_patches + prompt_updates
   - スキップ: Step 3全体（HF Processor: resize, normalize, PaS crop, tokenize等）
   - 4種実装: ProcessorOnlyCache / SenderCache / ShmSenderCache / ReceiverCache

2. **EncoderCache**（GPU, P1）
   - キー: mm_feature.identifier (= mm_hash or lora:mm_hash)
   - 保存: SiglipVisionModel + Projector出力テンソル (GPU)
   - スキップ: Step 4+5+6（ViT 27層 + Projector）
   - 論理管理: EncoderCacheManager (RefCount + FIFO)

3. **KVプレフィックスキャッシュ**（GPU, P1）
   - キー: hash(parent_hash, token_ids, extra_keys) — extra_keysにidentifier含む
   - 保存: デコーダ62層のKV状態ブロック
   - スキップ: Step 7+8の一部（プレフィックス一致分のデコーダforward）

### 重要な発見

- 3つのキャッシュは共通の mm_hash（blake3ハッシュ）を基盤として共有
- 各キャッシュは独立に動作（ヒット/ミスの組み合わせは理論上8通り）
- ProcessorCacheのヒットはIPC転送量も削減（Sender/Shmタイプ）

## 成果物

- `docs/src/investigations/gemma3-vision-caches.md` 新規作成
- `.state/questions.md` — 「MMプレフィックスキャッシュとProcessorCacheの相互作用」を解決済み

## 読んだソースコード

- `target/vllm/vllm/multimodal/hasher.py:50-162` — MultiModalHasher全体
- `target/vllm/vllm/multimodal/processing/processor.py:1299-1596` — ハッシュ計算〜キャッシュフロー
- `target/vllm/vllm/v1/core/kv_cache_utils.py:387-552` — MM extra keys + block hash
- `target/vllm/vllm/v1/core/encoder_cache_manager.py:91-117` — check_and_update_cache（既存知見で確認のみ）

## 次回へ

- GPUModelRunner 深堀り（最優先）
- KV Transfer / LMCache 調査（ユーザー関心2位）
