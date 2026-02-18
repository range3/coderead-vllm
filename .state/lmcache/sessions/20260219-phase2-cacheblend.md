# Phase 2 セッション2: CacheBlend MEDIUM化

**日付**: 2026-02-19
**フェーズ**: Phase 2
**所要時間**: 1セッション
**目標**: CacheBlend [SHALLOW] → [MEDIUM] 昇格、vLLMで動かす方法の調査

## 調査対象ファイル

| ファイル | 行数 | 内容 |
|---------|------|------|
| `lmcache/v1/compute/blend/blender.py` | 169 | LMCBlender（重要token同定・blend_layer Generator） |
| `lmcache/v1/compute/blend/metadata.py` | 35 | LMCBlendCommonMetadata/LMCBlendMetadata |
| `lmcache/v1/compute/blend/utils.py` | 64 | LMCBlenderBuilder（シングルトン的Blender生成） |
| `lmcache/v1/compute/models/base.py` | 142 | LMCBaseModel.compute_layer()（独自forward） |
| `lmcache/v1/compute/models/utils.py` | 69 | VLLMModelTracker、infer_model_from_vllm |
| `lmcache/v1/token_database.py` | 393+ | SegmentTokenDatabase |
| `lmcache/v1/multiprocess/blend_server.py` | 724 | BlendEngine（MPCacheEngine継承） |
| `lmcache/v1/config.py` | config section | Blending設定6項目 |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | 800+ | enable_blending処理 |
| `examples/blend_kv_v1/blend.py` | 218 | 使用例 |
| `examples/blend_kv_v1/README.md` | 18 | vLLMパッチ手順 |

## 主要発見

1. **blend_layerのGenerator同期**: `retrieve_layer`と`compute_layer`が各レイヤーで交互に進む
   - retrieve_layer: KVキャッシュをストレージ→GPUへロード
   - compute_layer: vLLMモデルの独自forward（layernorm→QKV→RoPE→blender→attn→MLP）
2. **重要token選択アルゴリズム**: check_layersのK差分L2ノルムtopk（`recomp_ratios[0]`割合）
3. **SegmentTokenDatabase**: セパレータで分割→各セグメント独立ハッシュ（プレフィックスチェーンではない）
4. **vLLMパッチ**: `load_model()`末尾に`VLLMModelTracker.register_model()`追加が必須
5. **BlendEngine**: BLEND_HASH_PREFIX=0xB1EDでプリコンピュートと通常キャッシュを区別
6. **対応モデル3種のみ**: Llama/Qwen2/Qwen3（`infer_model_from_vllm`でディスパッチ）

## 成果物

- `docs/src/lmcache/investigations/cacheblend.md` を新規作成

## 次回への引き継ぎ

- LookupClient/Server深堀りが最優先
- CacheBlend自体はTODO（TP対応、プレフィックスキャッシュ互換）が多くまだ開発中
