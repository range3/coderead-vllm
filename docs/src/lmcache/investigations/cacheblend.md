# CacheBlend: 非プレフィックスKVキャッシュ再利用

> **深度**: [MEDIUM]
> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-19（Phase 2 CacheBlend調査）

## 概要

CacheBlendは、**プレフィックスが一致しない場合でもKVキャッシュを再利用**する機能。通常のプレフィックスキャッシュはトークン列が同一プレフィックスで始まる場合のみ再利用できるが、CacheBlendはドキュメントの順序が変わっても再利用できる。

### 解決する問題

RAG（Retrieval Augmented Generation）などで複数ドキュメントをプロンプトに含める場合：
- 1回目: `[SYS] [SEP] [Doc A] [SEP] [Doc B] [SEP] [Doc C] [SEP] [質問1]`
- 2回目: `[SYS] [SEP] [Doc B] [SEP] [Doc A] [SEP] [Doc C] [SEP] [質問2]`

2回目はプレフィックスが変わるため通常のキャッシュは使えないが、CacheBlendはDoc A/B/Cそれぞれの事前計算KVキャッシュを再利用してblendする。

## アーキテクチャ全体図

```mermaid
graph TD
    subgraph "vLLM（パッチ必須）"
        GW[gpu_worker.py<br/>load_model()]
        VMT[VLLMModelTracker<br/>register_model()]
        GW --> VMT
    end

    subgraph "LMCacheConnectorV1Impl"
        SLK[start_load_kv<br/>blender.blend()]
    end

    subgraph "LMCBlender"
        BL[blend_layer()<br/>Generator]
        PQ[process_qkv()<br/>重要token同定]
    end

    subgraph "LMCBaseModel（compute_layer）"
        EMB[embedding]
        LN[layernorm]
        QKV[qkv_proj]
        ROPE[rotary_emb]
        ATT[flash_attn]
        MLP[mlp]
    end

    subgraph "LMCacheEngine"
        RL[retrieve_layer()<br/>Generator]
    end

    subgraph "Storage"
        CPU[LocalCPUBackend]
        DISK[LocalDiskBackend]
    end

    VMT --> |get_model| BL
    SLK --> BL
    BL --> |interleave| RL
    BL --> |interleave| EMB
    EMB --> LN --> QKV --> ROPE
    ROPE --> PQ
    PQ --> |重要tokenのみK/V更新| ATT
    ATT --> MLP
    RL --> CPU
    CPU --> DISK
```

## コンポーネント詳細

### 1. LMCBlender [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/compute/blend/blender.py:18`

CacheBlendのメイン制御クラス。

```python
class LMCBlender:
    def __init__(self, cache_engine, gpu_connector, vllm_model, config):
        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)
        self.num_layers = len(vllm_model.model.layers)
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )
```

#### blend_layer() — レイヤーワイズ処理のGenerator [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/compute/blend/blender.py:124`

```python
def blend_layer(self, tokens, mask=None, **kwargs):
    layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
    layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

    next(layerwise_retriever)  # 初期化
    yield

    for i in range(self.num_layers):
        next(layerwise_retriever)  # レイヤーiのKVキャッシュをGPUへロード
        next(layerwise_model_executor)  # レイヤーiのforward計算
        yield

    next(layerwise_retriever)  # 後処理
    self.metadata.clean()
    yield
```

**ポイント**: `retrieve_layer`（KVキャッシュロード）と`compute_layer`（forward計算）が各レイヤーで同期しながら交互に進む。KVキャッシュをロードしてからforward計算に使用。

#### process_qkv() — 重要token同定ロジック [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/compute/blend/blender.py:59`

check_layersで指定したレイヤーで重要tokenを決定する：

```python
def process_qkv(self, q, k, v, residual, layer_id, attn_output, attn_metadata):
    old_k, old_v = self.gpu_connector.get_kv(layer_id)

    # RoPE位置エンコーディング適用
    q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

    if layer_id in self.common_metadata.check_layers:
        # K差分のL2ノルム（token次元でsum）
        diff_k = torch.sum((k.to(float32) - old_k.to(float32)) ** 2, dim=[1])
        total_len = diff_k.shape[0]

        # recomp_ratios[0]の割合のtokenをtopk選択
        topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
        top_indices = torch.topk(diff_k, k=topk_num).indices
        top_indices, _ = torch.sort(top_indices)  # 順序保持

        # 重要tokenのみ選択してforward継続
        k, v = k[top_indices], v[top_indices]
        q = q[top_indices]
        self.metadata.imp_indices = top_indices

    if self.metadata.imp_indices is not None:
        # 重要tokenのみold_k/vを更新
        old_k[self.metadata.imp_indices] = k
        old_v[self.metadata.imp_indices] = v
        return q, old_k, old_v, ...  # 完全なK/V（重要token更新済み）
    else:
        return q, k, v, ...
```

**アルゴリズム**:
1. キャッシュから取得した `old_k` と新たに計算した `k` の差分L2ノルムを計算
2. 差分が大きい（= キャッシュが不正確）tokenを `recomp_ratios` 割合だけtopk選択
3. 重要tokenのみQ/K/Vを保持して再計算（他はキャッシュ値を使用）
4. 最終的に完全なK/V（重要token部分は更新済み）でAttentionを計算

### 2. LMCBaseModel.compute_layer() — モデルforward [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/compute/models/base.py:66`

`@torch.compile`デコレータが付いた独自forwardループ。vLLMの推論エンジンを迂回してLMCache独自の計算グラフを構築。

```python
@torch.compile
def compute_layer(self, input_ids):
    hidden_states = self.vllm_model.get_input_embeddings(input_ids)
    for idx, layer in enumerate(self.vllm_model.model.layers[...]):
        # QKV投影
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # モデル固有QKV処理（GQA等）
        q, k, v = self._process_qkv(q, k, v, layer)

        # LMCBlenderのprocess_qkv呼び出し（重要token選択）
        q, k, v, residual, attn_output, attn_metadata = \
            self.blender.process_qkv(q, k, v, residual, idx, ...)

        # Attention計算（重要tokenのみ）
        attn_output = self.lmc_attn_layers[idx].forward_contiguous(...)

        # MLP
        hidden_states = layer.mlp(hidden_states)

        yield  # 各レイヤー処理後にyield（blend_layer()と同期）
```

**対応モデル** (3種のみ):
- `LlamaForCausalLM` → `LMCLlamaModel`
- `Qwen2ForCausalLM` → `LMCLlamaModel`（同実装）
- `Qwen3ForCausalLM` → `LMCQwen3Model`

**参照**: `target/LMCache/lmcache/v1/compute/models/utils.py:14` (`infer_model_from_vllm`)

### 3. VLLMModelTracker — モデル参照管理 [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/compute/models/utils.py:38`

```python
class VLLMModelTracker:
    _vllm_models: Dict[str, nn.Module] = {}

    @classmethod
    def register_model(cls, instance_id: str, vllm_model: nn.Module): ...

    @classmethod
    def get_model(cls, instance_id: str) -> nn.Module: ...
```

クラス変数として全インスタンスで共有するシングルトン的レジストリ。`instance_id`は`ENGINE_NAME`（LMCacheの定数）が使われる。

### 4. LMCBlenderBuilder — ブレンダー生成 [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/compute/blend/utils.py:22`

```python
class LMCBlenderBuilder:
    @classmethod
    def get_or_create(cls, instance_id, cache_engine, gpu_connector, config):
        if instance_id not in cls._blenders:
            vllm_model = VLLMModelTracker.get_model(instance_id)
            blender = LMCBlender(cache_engine, gpu_connector, vllm_model, config)
            cls._blenders[instance_id] = blender
        return cls._blenders[instance_id]
```

### 5. SegmentTokenDatabase — セグメント単位ハッシュ [VERIFIED]

**参照**: `target/LMCache/lmcache/v1/token_database.py:393`

CacheBlend使用時はChunkedではなくSegmentTokenDatabaseを使用。

```python
class SegmentTokenDatabase(TokenDatabase):
    def __init__(self, config, metadata):
        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)
        self.sep_tokens = tokenizer.encode(config.blend_special_str)[1:]  # [1:]でBOS除去
        self.sep_len = len(self.sep_tokens)

    def _fast_split_by_subtensor(self, tokens):
        """スライディングウィンドウでsep_tokensを検索して分割"""
        windows = tokens.unfold(0, self.sep_len, 1)
        matches = (windows == self.sep_tokens).all(dim=1).nonzero(...)
        # マッチ位置でtokensを分割してyield

    def process_tokens(self, tokens, ...):
        """各セグメントごとに独立したハッシュを生成（プレフィックスチェーンではない）"""
        for token_chunk in self._fast_split_by_subtensor(tokens):
            yield (start_idx, end_idx, self._make_key_by_hash(self._hash_tokens(token_chunk)))
```

**ChunkedTokenDatabaseとの違い**:
- Chunked: 全トークンのプレフィックスハッシュチェーン（順序依存）
- Segment: セパレータで分割した各セグメントを独立ハッシュ（順序非依存）

## vLLMで動かす方法

### 必要なパッチ（vLLM本体への変更）

**参照**: `target/LMCache/examples/blend_kv_v1/README.md`

`target/vllm/vllm/v1/worker/gpu_worker.py`の`load_model()`末尾に追加：

```python
# load_model()の末尾（self.model_runner.load_model()の後）
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.integration.vllm.utils import ENGINE_NAME

VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
ensure_kv_transfer_initialized(self.vllm_config)
```

**なぜ必要か**: LMCacheのforward計算（`compute_layer`）がvLLMモデルの`.model.layers[]`に直接アクセスするため、実行時にvLLMモデルの参照が必要。KV Transferは`initialize_from_config()`で初期化されるため順序に注意。

> **注意**: READMEでは`init_worker_distributed_environment`内の`ensure_kv_transfer_initialized`をコメントアウトと記載。ただし最新vLLMでは同関数は`initialize_from_config()`内に移動済みのため、パッチ内容は使用するvLLMバージョンに依存する。

### 環境変数による設定

**参照**: `target/LMCache/examples/blend_kv_v1/blend.py:20`

```bash
# 基本設定
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_USE_LAYERWISE=True       # CacheBlendにはlayerwiseが必須

# Blending設定
export LMCACHE_ENABLE_BLENDING=True
export LMCACHE_BLEND_SPECIAL_STR=" # # "    # セパレータ文字列
export LMCACHE_BLEND_CHECK_LAYERS=1         # 重要token判定レイヤー（レイヤー1で判定）
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15  # 再計算するtoken割合（15%）

# ストレージ（CPU）
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5  # GB

# スパースアテンション（任意、FLASHINFERが必要）
export VLLM_ATTENTION_BACKEND=FLASHINFER
export LMCACHE_EXTRA_CONFIG='{"enable_sparse": true}'
```

### Pythonコード

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
)

llm_args = EngineArgs(
    model="meta-llama/Llama-2-7b-chat-hf",   # Llama/Qwen2/Qwen3のみ対応
    kv_transfer_config=ktc,
    enable_prefix_caching=False,  # 必須: CacheBlendと非互換
    enforce_eager=True,            # 必須: CUDAGraphはCacheBlendと非互換
    max_model_len=32768,
    gpu_memory_utilization=0.7,
)

llm = LLM(**asdict(llm_args))

# プロンプト構築: セパレータでセグメントを区切る
sep = tokenizer.encode(" # # ")[1:]   # [1:]でBOS除去
prompt = sys_tokens + sep + doc_a + sep + doc_b + sep + query_tokens

# 後処理
LMCacheEngineBuilder.destroy(ENGINE_NAME)
```

### プロンプト設計のポイント

CacheBlendが効果を発揮するプロンプト構造：
```
[SYS_PROMPT] [SEP] [Document_A] [SEP] [Document_B] [SEP] [Document_C] [SEP] [QUERY]
```

- 各セグメントがセパレータ(`LMCACHE_BLEND_SPECIAL_STR`)で区切られる
- セグメントの順序が変わっても各セグメントのKVキャッシュを再利用可能
- セグメントはchunk_size（256トークン）の倍数に揃えると効率的

## BlendEngine（MultiProcessモード）

**参照**: `target/LMCache/lmcache/v1/multiprocess/blend_server.py:98`

MultiProcessモードのCacheBlend用サーバー。`MPCacheEngine`を継承し、セパレータベースの段落分割・プリコンピュート保存・取得を提供。

```python
class BlendEngine(MPCacheEngine):
    BLEND_HASH_PREFIX = 0xB1ED  # 通常キャッシュとBlendキャッシュを区別するプレフィックス

    def __init__(self, sep_tokens, storage_manager_config, chunk_size=256):
        super().__init__(storage_manager_config, chunk_size, hash_algorithm="blake3")
        self._token_matcher = ParallelPatternMatcher(sep_tokens)  # C拡張による高速マッチング
```

### 主要メソッド

| メソッド | 役割 |
|---------|------|
| `cb_register_kv_cache` | GPUバッファ（KVキャッシュ）を登録 |
| `cb_lookup_pre_computed` | 事前計算済みチャンクのlookup（各段落ごとにprefetch） |
| `cb_store_pre_computed` | 事前計算済みチャンクをストレージに保存（`BLEND_HASH_PREFIX`でハッシュ計算） |
| `cb_retrieve_pre_computed` | ストレージからGPUバッファへKVキャッシュをコピー |
| `cb_store_final` | 最終KVキャッシュを通常ハッシュで保存（通常モードLLMでも利用可能に） |

**ハッシュ区別**: `BLEND_HASH_PREFIX=0xB1ED`でプリコンピュートキャッシュと通常キャッシュを区別。

## 設定パラメータ一覧

**参照**: `target/LMCache/lmcache/v1/config.py:100`

| 環境変数 | Pythonキー | デフォルト | 説明 |
|---------|-----------|---------|------|
| `LMCACHE_ENABLE_BLENDING` | `enable_blending` | `False` | CacheBlend有効化 |
| `LMCACHE_BLEND_SPECIAL_STR` | `blend_special_str` | `" # # "` | セパレータ文字列 |
| `LMCACHE_BLEND_CHECK_LAYERS` | `blend_check_layers` | `None` | 重要token判定レイヤー（カンマ区切りリスト） |
| `LMCACHE_BLEND_RECOMPUTE_RATIOS` | `blend_recompute_ratios` | `None` | 再計算割合（カンマ区切りfloatリスト） |
| `LMCACHE_BLEND_THRESHOLDS` | `blend_thresholds` | `None` | 重要token判定閾値（未使用/TODO） |
| `LMCACHE_BLEND_MIN_TOKENS` | `blend_min_tokens` | `256` | Blend対象の最小トークン数 |
| `LMCACHE_USE_LAYERWISE` | `use_layerwise` | `False` | レイヤーワイズ転送（CacheBlendには必須） |

**注意**: `enable_blending=True`にすると`save_unfull_chunk=True`が自動設定される（不完全チャンクも保存必要）。

## 制約・注意事項

### 対応モデル
- ✅ `LlamaForCausalLM` (Llama 2/3系)
- ✅ `Qwen2ForCausalLM` (Qwen2系)
- ✅ `Qwen3ForCausalLM` (Qwen3系)
- ❌ その他モデル（`NotImplementedError`）

### 非互換機能
- ❌ `enable_prefix_caching=True`（TODO: 対応予定コメントあり）
- ❌ CUDAGraph（`enforce_eager=True`が必要）

### 既知のTODO
- `recomp_ratios[0]`しか使わない（複数比率対応TODO）
- 異なるレイヤーで異なる比率をサポートするTODO
- 閾値ベースのblendingは未実装
- TP（テンソル並列）、PP、マルチモーダル未サポートのTODO

## 依存関係

```
LMCacheConnectorV1Impl.start_load_kv()
    └── LMCBlender.blend()
            └── blend_layer() [Generator]
                    ├── LMCacheEngine.retrieve_layer() [Generator] ← ストレージからGPUへKV転送
                    └── LMCBaseModel.compute_layer() [Generator] ← vLLMモデルの独自forward
                            └── LMCBlender.process_qkv() ← 重要token選択・KV更新

VLLMModelTracker.register_model() ← vLLMパッチ（load_model()末尾）
    └── LMCBlenderBuilder.get_or_create() ← blender初期化時に参照
```
