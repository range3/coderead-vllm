# LMCacheEngine

> **深度**: [MEDIUM] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 1 セッション1）

## 概要

LMCacheのコアコンポーネント（1,949行）。TokenDatabase、GPUConnector、StorageManagerを統合し、
KVキャッシュのstore/retrieveオペレーションを実行する。

**参照**: `target/LMCache/lmcache/v1/cache_engine.py`

## Store API（2つのエントリポイント）

### store_layer() — レイヤーワイズ（主要パス）

**参照**: `target/LMCache/lmcache/v1/cache_engine.py:528`

```python
def store_layer(
    tokens: Union[Tensor, list[int]],
    mask: Optional[Tensor] = None,
    **kwargs,  # kvcaches, slot_mapping, offset, sync, req_id
) -> Generator[None, None, None]
```

**Generator関数**。呼び出し側が各attentionレイヤー実行後に`next()`で進める。

**初期化フェーズ（最初のyieldまで）**:
1. `TokenDatabase.process_tokens(tokens, mask)` → `(start, end, CacheEngineKey)`のイテラブル
2. `key.split_layers(num_layers)` → `LayerCacheEngineKey`のリスト
3. `StorageManager.contains(keys[0])` で既存チェック（layer 0のキーで判定）
4. `StorageManager.batched_allocate(shape, dtype, batch_size=num_layers)` でMemoryObj確保
5. チャンク×レイヤー → レイヤー×チャンクに転置
6. `GPUConnector.batched_from_gpu(memory_objs, starts, ends, ...)` でGPU転送Generator生成

**レイヤーループ（num_layers回yield）**:
```
yield → next(mem_obj_generator) → StorageManager.batched_put(keys[layer_id], memory_objs[layer_id])
```

**エラーハンドリング**:
- `batched_allocate`がNone → メモリ不足、storeを中止（yieldだけ行う）
- `is_healthy()` False → 全操作スキップ
- `is_frozen()` True → freeze mode、yieldだけ行う

### store() — 非レイヤーワイズ

**参照**: `target/LMCache/lmcache/v1/cache_engine.py:335`

全レイヤー一括転送。`GPUConnector.from_gpu()`で全レイヤーをまとめてコピーし、`StorageManager.batched_put()`で保存。レイヤーワイズが無効の場合に使用。

## 主要な内部状態

| フィールド | 型 | 説明 |
|---|---|---|
| `token_database` | ChunkedTokenDatabase | トークン→チャンクハッシュ変換 |
| `gpu_connector` | GPUConnectorInterface | GPU↔CPU転送 |
| `storage_manager` | StorageManager | 多段バックエンド管理 |
| `num_layers` | int | モデルのレイヤー数 |
| `metadata` | LMCacheMetadata | model_name, world_size等 |
| `fmt` | MemoryFormat | KV_T2D or KV_2LTD |
| `kv_events` | list | BlockStored等のイベントキュー |

## 上流・下流

- **上流**: LMCacheConnectorV1Impl（store_layer/retrieve呼び出し）
- **下流**: TokenDatabase、GPUConnector、StorageManager
- **ライフサイクル**: LMCacheManagerが生成・管理
