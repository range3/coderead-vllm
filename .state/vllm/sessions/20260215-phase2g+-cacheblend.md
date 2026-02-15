# Phase 2g+: CacheBlend 実装調査

**日付**: 2026-02-15
**フェーズ**: Phase 2 (KV Transfer DEEP化の一環)
**テーマ**: CacheBlendの実装詳細とvLLMプラグイン境界

## 調査範囲

- LMCache `v1/compute/blend/` — Blenderコア
- LMCache `v1/compute/models/` — 独自forward path
- LMCache `v1/compute/attention/` — 独自attentionバックエンド
- LMCache `v1/gpu_connector/gpu_connectors.py` — VLLMBufferLayerwiseGPUConnector
- LMCache `v1/multiprocess/blend_server.py` — BlendServer
- LMCache `integration/vllm/vllm_v1_adapter.py` — blending統合
- vLLM `lmcache_integration/vllm_v1_adapter.py` — native側blending
- LMCache `examples/blend_kv_v1/README.md` — ad-hocパッチ記載

## 主要発見

### 1. vLLM本体パッチが必須

`examples/blend_kv_v1/README.md`に明記。`gpu_worker.py`の`load_model()`末尾に`VLLMModelTracker.register_model()`追加 + `init_worker_distributed_environment()`の`ensure_kv_transfer_initialized()`をコメントアウト → `load_model()`末尾に移動。

### 2. 独自forward path

CacheBlendはvLLMの通常のforward pathをバイパス。`LMCBaseModel.compute_layer()`はvLLMモデルの`.model.layers[i]`に直接アクセスして、embedding→layernorm→QKV投影→process_qkv→blending→attention→output投影→MLPを独自実行。`@torch.compile`でコンパイルされるがCUDAGraphは未使用。

### 3. 重要token同定

`LMCBlender.process_qkv()`で実行。`check_layers`で指定されたレイヤーで、新旧KのL2距離のtopkを選択。`recomp_ratios[0]`の割合（例: 0.1 = 10%）。一度同定されたインデックスは以降の全レイヤーで再利用。

### 4. VLLMBufferLayerwiseGPUConnector

blending専用GPUコネクタ。通常のLayerwiseと異なり:
- 中間GPUバッファに保持（paged memoryに直接書かずblenderからアクセス可能に）
- FusedRope（カスタムCUDAカーネル）でRoPE位置補正
- ギャップゼロイング（RAGチャンク間セパレータ位置）
- ダブルバッファでロード/計算パイプライン

### 5. 対応モデル・制約

- Llama, Qwen2, Qwen3の3モデルのみ
- RoPE: rotary_dim==head_size、scaling無し、factor=1.0のみ
- TP/PP未対応、プレフィックスキャッシュ非互換、バッチサイズ1前提
- レイヤー別ratio/閾値ベースblendingはTODO

### 6. BlendServer

`MPCacheEngine`継承。セパレータトークンで段落分割（`ParallelPatternMatcher` C実装）。`BLEND_HASH_PREFIX = 0xB1ED`でblend専用ハッシュ。store_final()は通常ハッシュで保存し通常モードLLMからも利用可能。

### 7. プラグイン境界の問題

KVConnectorBase_V1のAPIはKV読み書きのみ想定。CacheBlendの「独自forward＋選択的再計算」はスコープ外。モデルオブジェクトへの参照がメタデータで渡せない。

## 成果物

- `docs/src/investigations/cacheblend-implementation.md` 新規作成
- `docs/src/investigations/lmcache-integration.md` CacheBlendセクション追加
- `.state/questions.md` CacheBlend疑問を解決済みに
- `.state/context-index.md` 新規ドキュメント追加

## 未調査

- BlendServer経由のマルチプロセスblending（`blend_server.py`の統合部分詳細）
- `SegmentTokenDatabase`の動作（`ChunkedTokenDatabase`との差分）
- `flash_infer_sparse.py`の`HackBSAWrapper`の完全な動作
