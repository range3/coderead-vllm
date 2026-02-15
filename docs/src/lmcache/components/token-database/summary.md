# TokenDatabase（ChunkedTokenDatabase）

> **深度**: [MEDIUM] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 1 セッション1）

## 概要

トークン列をチャンクに分割し、プレフィックスチェーンハッシュを計算してCacheEngineKeyを生成する。
vLLMのプレフィックスキャッシュと**同一のハッシュアルゴリズム**を使用し、キーの互換性を保証する。

**参照**: `target/LMCache/lmcache/v1/token_database.py`

## クラス階層

```
TokenDatabase (ABC)
  ├── ChunkedTokenDatabase    ← 標準実装（固定サイズチャンク）
  └── SegmentTokenDatabase    ← CacheBlend用（セパレータベースの可変長チャンク）
```

## 主要メソッド

### process_tokens()

**参照**: `target/LMCache/lmcache/v1/token_database.py:309`

```python
def process_tokens(
    tokens: Optional[Union[Tensor, List[int]]] = None,
    hashes: Optional[List[int]] = None,
    offsets: Optional[List[int]] = None,
    mask: Optional[Tensor] = None,
    make_key: bool = True,
    request_configs: Optional[dict] = None,
) -> Iterable[ProcessTokensResult]  # (start, end, CacheEngineKey|hash)
```

**2つの入力モード**:
1. **tokens入力**: トークン列を受け取り、チャンク分割→ハッシュ計算
2. **hashes入力**: 事前計算済みハッシュ+offsetsを受け取り、キー生成のみ

**チャンク分割アルゴリズム**（tokens入力時）:
1. `_chunk_tokens()`: chunk_size（デフォルト256）単位で分割
   - `save_unfull_chunk=True`（デフォルト）: 端数チャンクも保存
   - `save_unfull_chunk=False`: 端数は切り捨て
2. `_prefix_hash()`: プレフィックスチェーンハッシュを計算
   - 初期値: `NONE_HASH`（vLLMから取得、`kv_cache_utils.init_none_hash()`）
   - 各チャンク: `hash_func((previous_hash, token_tuple, extra_keys))`
3. maskのFalse区間（=already-cached prefix）のチャンクをスキップ
   - **制約**: False数はchunk_sizeの倍数でなければならない（ValueError）

### ハッシュ関数

**参照**: `target/LMCache/lmcache/v1/token_database.py:97` (`_get_vllm_hash_func`)

vLLMの`get_hash_fn_by_name("sha256_cbor")`を直接利用。
複数のインポートパスを試行し、vLLMバージョン互換性を確保:
- `vllm.utils.hashing.get_hash_fn_by_name`（PR#27151以降）
- `vllm.utils.get_hash_fn_by_name`（PR#27151以前）
- `sha256_cbor_64bit`→`sha256_cbor`リネーム対応（PR#23673）
- フォールバック: Python組み込み`hash()`（非推奨、分散キャッシュで不整合の可能性）

### CacheEngineKey生成

**参照**: `target/LMCache/lmcache/v1/token_database.py:207` (`_make_key_by_hash`)

```python
CacheEngineKey(
    model_name,         # メタデータから
    world_size,         # save_only_first_rank時は1に固定
    worker_id,          # GPUランク
    chunk_hash,         # プレフィックスチェーンハッシュ
    kv_dtype,           # e.g. bfloat16
    request_configs,    # オプション（per-requestの設定dict）
)
```

`save_only_first_rank`はMLA（Multi-head Latent Attention）使用時に有効。world_sizeを1に固定することで、異なるTP並列度でもキーが一致する。

## 上流・下流

- **上流**: LMCacheEngine（store_layer/store/retrieve等で呼び出し）
- **下流**: なし（純粋な変換コンポーネント）
- **依存**: vLLMのハッシュ関数ライブラリ
