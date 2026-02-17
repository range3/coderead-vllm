# Phase 2h: mm_hash計算方法調査

> **日付**: 2026-02-17
> **Phase**: 2h（マルチモーダルDEEP化の一部）
> **対象OSS**: vLLM

## 目的

mm_hashの計算方法をGemma3モデルを例に追跡する。

## 調査内容

### 1. ハッシュ計算の3層構造

- **`hash_kwargs()`** (`hasher.py:154`): kwargsをキー名ソート→`iter_item_to_bytes()`で逐次バイト変換→hasher投入→hexdigest
- **`iter_item_to_bytes()`** (`hasher.py:134`): dict/list/tupleを再帰展開、キー名プレフィックス付きバイト列をyield
- **`serialize_item()`** (`hasher.py:52`): 型別シリアライズ（PIL Image、Tensor、ndarray等）

### 2. 画像の3つのシリアライズパス

1. **EXIF UUID高速パス**: ExifTags.ImageIDがUUID型の場合→16バイトのみ
2. **MediaWithBytes高速パス**: ラッパー保持時→original_bytes（JPEG/PNG等エンコード済み）
3. **ピクセルデータパス**: 通常のPIL Image→mode+np.asarray()で全ピクセル展開

`get_item_for_hash()`がMediaWithBytesラッパーを剥がさずに返すことで、パス2が有効になる。

### 3. `_hash_mm_items()`のmm_uuids分岐

- mm_uuidsなし → 画像データからhash_kwargs()計算
- mm_uuidsあり + kwargs非空 → UUID文字列をitemとしてhash_kwargs()に投入（高速化）
- mm_uuidsあり + kwargs空 → UUIDそのものをmm_hashとして使用（ハッシュ計算スキップ）

### 4. identifier vs mm_hash

- `mm_hash`: 純粋なコンテンツハッシュ（ProcessorCacheキー、LoRA非依存）
- `identifier`: LoRA対応で`{lora_name}:{mm_hash}`になりうる（EncoderCache/プレフィックスキャッシュキー）

### 5. プレフィックスキャッシュのextra_keys

`_gen_mm_extra_hash_keys()`がブロック範囲とMMプレースホルダ範囲の重なりを検出し、`mm_feature.identifier`をextra_keysに追加。

### 6. Gemma3固有の動作

Gemma3は`_hash_mm_items()`をオーバーライドしていない。デフォルト実装がそのまま使われる。PaSはハッシュ計算後に実行されるため、同じ画像は同じmm_hashを持つ。

## 成果物

- `docs/src/vllm/components/multimodal/mm-processing.md` §3を[MEDIUM]→[DEEP]昇格
  - 10サブセクション（3.1〜3.10）に詳細化
- `docs/src/vllm/components/multimodal/summary.md` の詳細ドキュメントテーブル更新

## 主要ソースファイル

| ファイル | 調査箇所 |
|---------|---------|
| `target/vllm/vllm/multimodal/hasher.py` | MultiModalHasher全体（163行、全行読了） |
| `target/vllm/vllm/multimodal/processing/processor.py` | `_hash_mm_items()` (L1300-1364)、`_cached_apply_hf_processor()` (L1514-) |
| `target/vllm/vllm/multimodal/parse.py` | `get_item_for_hash()` (L88-92, L118-120) |
| `target/vllm/vllm/v1/engine/input_processor.py` | `_get_mm_identifier()` (L273-289)、MultiModalFeatureSpec構築 (L413-437) |
| `target/vllm/vllm/v1/core/kv_cache_utils.py` | `_gen_mm_extra_hash_keys()` (L388-444) |
| `target/vllm/vllm/multimodal/inputs.py` | MultiModalFeatureSpec定義 (L337-367)、MultiModalHashes型 (L1053) |
| `target/vllm/vllm/envs.py` | VLLM_MM_HASHER_ALGORITHM (L73, L793) |
