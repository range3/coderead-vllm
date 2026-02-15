# Gemma3 27B ビジョンパイプライン: 形状フローと数値まとめ

## モデルパラメータ（config.json + preprocessor_config.json）

| パラメータ | 値 | 出典 |
|---|---|---|
| image_size | 896 | vision_config |
| patch_size | 14 | vision_config |
| vision hidden_size | 1152 | vision_config |
| vision num_heads | 16 | vision_config |
| vision num_layers | 27 | vision_config |
| text hidden_size | 5376 | text_config |
| text num_heads | 32 | text_config |
| text num_layers | 62 | text_config |
| mm_tokens_per_image | 256 | config.json |
| image_token_index | 262144 | config.json |
| boi_token_index | 255999 | config.json |
| eoi_token_index | 256000 | config.json |

### 導出値

| 導出パラメータ | 計算 | 値 |
|---|---|---|
| patches_per_image | 896 / 14 | **64** |
| エンコーダ入力パッチ数 | 64² | **4096** |
| tokens_per_side | √256 | **16** |
| AvgPool2d kernel_size | 64 / 16 | **4** |
| Projector 出力トークン/画像 | 16² | **256** (= mm_tokens_per_image ✅) |

---

## Pan-and-Scan 設定

| パラメータ | preprocessor_config.json | フォールバックデフォルト | 出典 |
|---|---|---|---|
| do_pan_and_scan | null | **False** | processing_gemma3.py L44 |
| pan_and_scan_min_crop_size | null | **256** | processing_gemma3.py L45 |
| pan_and_scan_max_num_crops | null | **4** | processing_gemma3.py L46 |
| pan_and_scan_min_ratio_to_activate | null | **1.2** | processing_gemma3.py L47 |

- Google はモデル配布時にこれらを**すべて null** にしている
- デフォルト値は HF transformers の `Gemma3ProcessorKwargs._defaults` で定義
- **デフォルトでは Pan-and-Scan は無効**
- 有効化: vLLM では `--hf-overrides '{"do_pan_and_scan": true}'`

---

## API リクエストからデコーダ入力までの全体フロー

### Step 1: ユーザーの API リクエスト

```json
{
  "model": "gemma-3-27b-it",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "この文書を要約して"}
      ]
    }
  ]
}
```

ユーザーは画像を1枚渡すだけ。クロップの存在を意識する必要はない。

### Step 2: chat_template 適用

vLLM が chat_template を適用してプロンプト文字列を生成:

```
<start_of_turn>user
<start_of_image>この文書を要約して<end_of_turn>
<start_of_turn>model
```

`<start_of_image>` は `boi_token` (token_id=255999)。この時点ではプレースホルダが **1個だけ**。

### Step 3: Gemma3Processor.__call__() — CPU 側前処理

#### 3a: image_processor による画像前処理

```python
image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
```

画像をリサイズ・正規化し、Pan-and-Scan が有効ならクロップも生成する。

#### 3b: num_crops の取得

```python
num_crops = to_py_obj(image_inputs.pop("num_crops"))
```

#### 3c: プロンプトの自動書き換え（Pan-and-Scan 時のみ）

```python
for num, idx in reversed(list(zip(num_crops, image_indexes))):
    if num:  # num=0 なら falsy → この書き換えは発生しない
        formatted_image_text = (
            f"Here is the original image {self.boi_token} "
            f"and here are some crops to help you see better "
            + " ".join([self.boi_token] * num)
        )
        prompt = prompt[:idx] + formatted_image_text + prompt[idx + len(self.boi_token):]
```

**Pan-and-Scan 無効（num=0）時**: `if num:` が falsy なので、書き換えは**一切発生しない**。
`<start_of_image>` は1個のまま次のステップへ。

**Pan-and-Scan 有効（num=2）時**: 1個の `<start_of_image>` が以下に置き換えられる:
```
Here is the original image <start_of_image> and here are some crops to help you see better <start_of_image> <start_of_image>
```

#### 3d: boi_token → full_image_sequence への展開

```python
self.full_image_sequence = f"\n\n{boi_token}{image_token * 256}{eoi_token}\n\n"
# = "\n\n<start_of_image><image>×256<end_of_image>\n\n"

text = [prompt.replace(self.boi_token, self.full_image_sequence) for prompt in text]
```

全ての `<start_of_image>` がそれぞれ 256個の `<image>` トークンを含む `full_image_sequence` に展開される。

#### 3e: tokenizer で token_ids に変換

```python
text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
```

`<image>` トークン (token_id=262144) が並んだ input_ids が生成される。

#### 3f: token_type_ids の生成

```python
mm_token_type_ids[array_ids == self.image_token_id] = 1
# → <image> トークン位置が 1、それ以外が 0
```

---

## ケース1: デフォルト（Pan-and-Scan 無効）

入力例: A4 150dpi 画像 (1240 × 1754 pixel)

### プロンプト変換の流れ

```
ユーザー入力:
  画像1枚 + "この文書を要約して"

chat_template 適用後:
  "...<start_of_image>この文書を要約して..."
                ↑
          boi_token 1個

do_pan_and_scan=False → num_crops=0 → プロンプト書き換えなし

boi_token → full_image_sequence 展開後:
  "...\n\n<start_of_image><image>×256<end_of_image>\n\nこの文書を要約して..."
           ↑              ↑×256  ↑
         255999         262144  256000

tokenize 後の input_ids (概念的):
  [..., 255999, 262144, 262144, ...(×256)..., 262144, 256000, ..., テキスト, ...]
```

### CPU 側前処理

```
元画像 (1240×1754)
    │  resize(896×896, bilinear)   ← アスペクト比無視の正方形リサイズ
    │  rescale(× 1/255)            ← [0,255] → [0,1]
    │  normalize(mean=0.5, std=0.5) ← [0,1] → [-1,1]
    ▼
pixel_values: (1, 3, 896, 896)
num_patches:  tensor([1])
```

### GPU 側: SiglipVisionModel

```
(1, 3, 896, 896)
    │  Conv2d(3 → 1152, kernel=14, stride=14)
    ▼
(1, 1152, 64, 64)              ← 896/14 = 64
    │  flatten + transpose
    ▼
(1, 4096, 1152)                ← 64² = 4096 パッチ
    │  + position_embedding(4096, 1152)
    ▼
(1, 4096, 1152)
    │  SiglipEncoder × 27層
    │  （双方向 Attention, heads=16, 4096トークン間全対全）
    ▼
(1, 4096, 1152)
    │  post_layernorm
    ▼
(1, 4096, 1152)
```

### GPU 側: Gemma3MultiModalProjector

```
(1, 4096, 1152)
    │  transpose → (1, 1152, 4096)
    │  reshape  → (1, 1152, 64, 64)     ← 2Dグリッドに復元
    │
    │  AvgPool2d(kernel_size=4, stride=4)
    ▼
(1, 1152, 16, 16)                       ← 64/4 = 16
    │  flatten(2) → (1, 1152, 256)
    │  transpose  → (1, 256, 1152)       ← 16² = 256 トークン
    │
    │  GemmaRMSNorm(1152)
    ▼
(1, 256, 1152)
    │
    │  matmul(mm_input_projection_weight)  ← shape: (1152, 5376)
    ▼
(1, 256, 5376)                           ← text hidden_size 空間
```

### GPU 側: split + flatten

```
(1, 256, 5376)
    │  split by num_patches=[1] → [(1, 256, 5376)]
    │  flatten(0, 1)
    ▼
(256, 5376)                              ← 最終出力
```

### GPU 側: テキスト埋め込みとマージ

```python
text_embeds = embed_tokens(input_ids) * normalizer   # (seq_len, 5376)
# token_id=262144 は vocab 外 → handle_oov_mm_token=True でゼロ埋め
# is_multimodal: (seq_len,) ← 256箇所が True

merged = masked_scatter_(text_embeds, is_multimodal, mm_embeds)  # (256, 5376)
# → 262144 だった256箇所をビジョン埋め込みで上書き
# ※ ビジョン埋め込みには normalizer スケーリングは適用されない

→ (seq_len, 5376) として Gemma3 Decoder (62層) へ
```

### 消費トークン数: **256**

---

## ケース2: Pan-and-Scan 有効

入力例: 同じ A4 150dpi 画像 (1240 × 1754 pixel)

### プロンプト変換の流れ

```
ユーザー入力:
  画像1枚 + "この文書を要約して"

chat_template 適用後:
  "...<start_of_image>この文書を要約して..."
                ↑
          boi_token 1個

do_pan_and_scan=True → ratio=1754/1240≈1.415 > 1.2 → num_crops=2

Processor がプロンプトを自動書き換え (Step 3c):
  "...Here is the original image <start_of_image> and here are
   some crops to help you see better <start_of_image> <start_of_image>
   この文書を要約して..."
                                      ↑               ↑              ↑
                                 original 用       crop 0 用      crop 1 用

boi_token → full_image_sequence 展開後:
  "...Here is the original image \n\n<boi><image>×256<eoi>\n\n and here are
   some crops to help you see better \n\n<boi><image>×256<eoi>\n\n
   \n\n<boi><image>×256<eoi>\n\nこの文書を要約して..."

tokenize 後:
  [..., "Here", "is", ...,
   255999, 262144×256, 256000,          ← original
   ..., "and", "here", ...,
   255999, 262144×256, 256000,          ← crop 0
   ...,
   255999, 262144×256, 256000,          ← crop 1
   ..., テキスト, ...]
```

### CPU 側: Pan-and-Scan 判定

```python
# 縦長画像 (height > width)
ratio = 1754 / 1240 ≈ 1.415
min_ratio_to_activate = 1.2
1.415 > 1.2 → ✅ 発動
```

### CPU 側: クロップ数計算

```python
# 縦長パス (image_height > image_width)
num_crops_h = min(
    floor(1754 / 256),          # = 6  ← min_crop_size 制約
    floor(1754 / 1240 + 0.5),   # = 1  ← アスペクト比近似
)
# → min(6, 1) = 1
num_crops_h = max(2, 1) = 2    # 最低2クロップに強制
num_crops_h = min(4, 2) = 2    # max_num_crops でクリップ
num_crops_w = 1

# クロップサイズ検証
crop_size_w = ceil(1240 / 1) = 1240
crop_size_h = ceil(1754 / 2) = 877
min(1240, 877) = 877 > 256 (min_crop_size) → ✅ 有効

結果: 1 × 2 = 2 クロップ
```

### CPU 側: クロップ切り出し + リサイズ

```
元画像 (1240×1754)
  ├── original  (1240×1754) → resize(896×896) → normalize → (3, 896, 896)
  ├── crop 0    (1240×877)  → resize(896×896) → normalize → (3, 896, 896)
  └── crop 1    (1240×877)  → resize(896×896) → normalize → (3, 896, 896)
                                                              │ stack
                                                pixel_values: (3, 3, 896, 896)
                                                num_patches:  tensor([3])
```

### GPU 側: SiglipVisionModel

```
(3, 3, 896, 896)
    │  Conv2d(3 → 1152, kernel=14, stride=14)
    ▼
(3, 1152, 64, 64)
    │  flatten + transpose
    ▼
(3, 4096, 1152)                 ← 3枚 × 4096パッチ
    │  + position_embedding
    ▼
(3, 4096, 1152)
    │  SiglipEncoder × 27層（双方向 Attention）
    ▼
(3, 4096, 1152)
```

### GPU 側: Gemma3MultiModalProjector

```
(3, 4096, 1152)
    │  → reshape → (3, 1152, 64, 64)
    │  AvgPool2d(k=4, s=4)
    ▼
(3, 1152, 16, 16)
    │  flatten + transpose
    ▼
(3, 256, 1152)
    │  RMSNorm → matmul(1152 → 5376)
    ▼
(3, 256, 5376)
```

### GPU 側: split + flatten

```
(3, 256, 5376)
    │  split by num_patches=[3] → [(3, 256, 5376)]
    │  flatten(0, 1)
    ▼
(768, 5376)                     ← 3 × 256 = 768 トークン
```

### GPU 側: テキスト埋め込みとマージ

```
input_ids 中の token_id=262144 が768箇所
↓ masked_scatter_ で (768, 5376) を順番に書き込み
→ (seq_len, 5376) として Gemma3 Decoder へ
```

### 消費トークン数: **768** (= 256 × 3)

---

## プロンプト比較

| | Pan-and-Scan 無効（デフォルト） | Pan-and-Scan 有効 |
|---|---|---|
| プロンプト書き換え | なし | "Here is the original image ... crops ..." 挿入 |
| boi_token 数 | 1 | 1 + num_crops (= 3) |
| `<image>` トークン数 | 256 | 256 × (1 + num_crops) = 768 |
| 装飾テキスト | なし | "Here is the original image", "and here are some crops to help you see better" |
| pixel_values shape | (1, 3, 896, 896) | (3, 3, 896, 896) |
| num_patches | tensor([1]) | tensor([3]) |

---

## 全体データフロー図

```
                        ┌─────────────────────────────┐
                        │  OpenAI 互換 API リクエスト    │
                        │  画像1枚 + テキスト           │
                        └────────────┬────────────────┘
                                     │
                        ┌────────────┴────────────────┐
                        │  chat_template 適用           │
                        │  → "<start_of_image>テキスト"  │
                        │    boi_token(255999) が1個    │
                        └────────────┬────────────────┘
                                     │
                        ┌────────────┴────────────────┐
                        │  CPU: Gemma3Processor        │
                        │                              │
                        │  image_processor:             │
                        │    resize(896×896)            │
                        │    rescale(×1/255)            │
                        │    normalize(0.5, 0.5)        │
                        │    Pan-and-Scan 時はクロップ生成│
                        │                              │
                        │  do_pan_and_scan?             │
                        │  ├── False:                   │
                        │  │   書き換えなし              │
                        │  │   boi_token 1個のまま       │
                        │  │   pixel_values: (1,3,896,896)│
                        │  │                            │
                        │  └── True & ratio > 1.2:      │
                        │      "Here is the original    │
                        │       image <boi> and here    │
                        │       are some crops ...      │
                        │       <boi> <boi>"            │
                        │      boi_token 3個に増加       │
                        │      pixel_values: (3,3,896,896)│
                        │                              │
                        │  各 boi_token を展開:          │
                        │  "\n\n<boi><img>×256<eoi>\n\n" │
                        │                              │
                        │  tokenizer → input_ids        │
                        │  token_type_ids 生成           │
                        └────────────┬────────────────┘
                                     │
                        input_ids:     [..., 262144×256, ...(×N)...]
                        pixel_values:  (total_patches, 3, 896, 896)
                        num_patches:   (num_images,)
                                     │
                        ═════════════╪═══════════════ CPU → GPU
                                     │
                        ┌────────────┴────────────────┐
                        │  GPU: SiglipVisionEmbeddings │
                        │  Conv2d(3→1152, k=14, s=14)  │
                        │  + position_embedding         │
                        └────────────┬────────────────┘
                                     │
                        (total_patches, 4096, 1152)
                                     │
                        ┌────────────┴────────────────┐
                        │  GPU: SiglipEncoder           │
                        │  27層 双方向 Transformer       │
                        │  heads=16, hidden=1152        │
                        └────────────┬────────────────┘
                                     │
                        (total_patches, 4096, 1152)
                                     │
                        ┌────────────┴────────────────┐
                        │  GPU: Gemma3MultiModalProjector│
                        │  reshape → (*, 1152, 64, 64) │
                        │  AvgPool2d(k=4, s=4)          │
                        │  → (*, 1152, 16, 16)          │
                        │  flatten + transpose           │
                        │  → (*, 256, 1152)             │
                        │  GemmaRMSNorm(1152)            │
                        │  matmul(1152 → 5376)           │
                        └────────────┬────────────────┘
                                     │
                        (total_patches, 256, 5376)
                                     │
                        ┌────────────┴────────────────┐
                        │  split by num_patches         │
                        │  flatten(0, 1) per image      │
                        │  → list[(N×256, 5376)]        │
                        └────────────┬────────────────┘
                                     │
                        ┌────────────┴────────────────┐
                        │  GPU: embed_input_ids()       │
                        │                              │
                        │  text = embed_tokens(ids)     │
                        │         × normalizer          │
                        │  ※ 262144 は vocab 外         │
                        │    → handle_oov_mm_token=True │
                        │    → ゼロ埋め                  │
                        │                              │
                        │  masked_scatter_(             │
                        │    text, is_multimodal,       │
                        │    mm_embeds)                 │
                        │                              │
                        │  ※ vision embeds には          │
                        │    normalizer 未適用            │
                        └────────────┬────────────────┘
                                     │
                        (seq_len, 5376)
                                     │
                        ┌────────────┴────────────────┐
                        │  GPU: Gemma3 Decoder          │
                        │  62層, heads=32, kv_heads=16  │
                        │  sliding_window=1024          │
                        │  head_dim=128                 │
                        └──────────────────────────────┘
```

---

## 注意事項

1. **ビジョン埋め込みの正規化**: テキスト埋め込みには `embed_tokens(ids) × normalizer` のスケーリングが適用されるが、ビジョン埋め込みには `mm_soft_emb_norm`（RMSNorm）のみが適用され、`normalizer` スケーリングは適用されない。

2. **V1 での制限**: Pan-and-Scan 有効時、V1 エンジンでは画像トークン間の双方向アテンションが簡略化されたパターンで実装されており、元モデルのアテンションパターンと完全には一致しない。

3. **AvgPool2d の役割**: エンコーダは 4096 パッチ（64×64 グリッド）の高解像度で処理しつつ、AvgPool2d(k=4, s=4) で 256 トークン（16×16）に圧縮して LLM に渡す。これにより計算量と情報量のバランスを取っている。

4. **Pan-and-Scan のプロンプト**: クロップありの場合のみ、Processor が "Here is the original image ... and here are some crops to help you see better ..." という装飾テキストを自動挿入する。クロップなしの場合この装飾テキストは存在せず、`<image>` トークン列のみとなる。ユーザーはクロップの存在を意識する必要はない。

5. **token_id=262144 の扱い**: `<image>` トークンの token_id=262144 は通常の vocab 範囲外（OOV）。`handle_oov_mm_token=True` により安全にゼロ埋めされ、後続の `masked_scatter_` でビジョン埋め込みに上書きされる。

6. **Pan-and-Scan のデフォルト値の出典**: `min_ratio_to_activate=1.2` 等の値は Google がモデルと共に配布した設定ではなく（preprocessor_config.json では全て null）、HF transformers の `processing_gemma3.py` 内の `Gemma3ProcessorKwargs._defaults` にハードコードされたフォールバック値。

---

## 主要ファイル参照

| ファイル | 主要クラス/関数 |
|---|---|
| vllm/.../gemma3_mm.py | Gemma3ForConditionalGeneration, Gemma3MultiModalProjector, Gemma3ProcessingInfo |
| vllm/.../siglip.py | SiglipVisionModel, SiglipVisionEmbeddings, SiglipEncoder |
| vllm/.../utils.py | _merge_multimodal_embeddings() |
| HF transformers/.../processing_gemma3.py | Gemma3Processor, Gemma3ProcessorKwargs (デフォルト値定義) |
| HF transformers/.../image_processing_gemma3.py | Gemma3ImageProcessor (Pan-and-Scan 実装) |
