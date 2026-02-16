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

## Retrieve API（2つのエントリポイント）

### retrieve() — Bulk（デフォルト）

**参照**: `target/LMCache/lmcache/v1/cache_engine.py:708`

```python
def retrieve(
    tokens: Union[Tensor, list[int]],
    mask: Optional[Tensor] = None,
    **kwargs,  # kvcaches, slot_mapping, request_configs, req_id
) -> torch.Tensor  # ret_mask (bool, CPU)
```

全レイヤーのKVキャッシュを一括取得し、GPUのページドメモリに書き戻す。

**処理フロー**:
1. `_process_tokens_internal()`（同期）or `_async_process_tokens_internal()`（非同期prefetch済み）でMemoryObjを取得
2. `save_only_first_rank`時は`_broadcast_or_receive_memory_objs()`で他ランクにブロードキャスト
3. `GPUConnector.batched_to_gpu(memory_objs, starts, ends, ...)`で一括GPU転送
4. `memory_obj.ref_count_down()`で解放
5. `remove_after_retrieve`時は`StorageManager.remove(key)`で即座に削除

**_process_tokens_internal()**（同期パス）:
1. `process_tokens()`でチャンク分割
2. `get_block_mapping()`でチャンクの所在バックエンドをprefix matchで特定
3. `batched_get(keys, location)`でバックエンドからMemoryObj取得
4. 取得失敗時は`last_failed_block_start`以降を全て無効化

**_async_process_tokens_internal()**（非同期パス）:
1. `event_manager.pop_event(LOADING, req_id)`でprefetch済みFutureを取得
2. `future.result()`でtier×chunkのMemoryObjマップを構築
3. `process_tokens()`で再チャンク分割しマッチング
4. 未使用MemoryObjは即座に`ref_count_down()`

### retrieve_layer() — Layerwise

**参照**: `target/LMCache/lmcache/v1/cache_engine.py:851`

```python
def retrieve_layer(
    tokens: Union[Tensor, list[int]],
    mask: Optional[Tensor] = None,
    **kwargs,  # kvcaches, slot_mapping, sync
) -> Generator[Optional[Tensor], None, None]
```

レイヤー単位でKVキャッシュを取得するGenerator関数。

**初期化フェーズ**:
1. `process_tokens()`でチャンク分割
2. `StorageManager.contains(layer0_key)`でヒット＋location統一チェック
3. キーをlayer-major形式に転置: `keys[chunk][layer]` → `keys_layer_major[layer][chunk]`
4. `StorageManager.layerwise_batched_get(keys_layer_major, location)` → `get_generator`
5. `GPUConnector.batched_to_gpu(starts, ends, ...)` → `mem_obj_consumer` Generator

**レイヤーループ**:
```
yield → task = next(get_generator) → mem_objs = task.result() → mem_obj_consumer.send(mem_objs)
```

最終yield時に`ret_mask`を返す。`ref_count_down()`は全レイヤー完了後にバッチ実行。

### lookup() — ヒット数問い合わせ

**参照**: `target/LMCache/lmcache/v1/cache_engine.py:992`

Scheduler側から呼ばれるヒット数チェック。`process_tokens()`でチャンク分割し、`StorageManager`の`contains()`/`batched_contains()`でプレフィックスマッチ。

## 上流・下流

- **上流**: LMCacheConnectorV1Impl（store_layer/retrieve呼び出し）
- **下流**: TokenDatabase、GPUConnector、StorageManager
- **ライフサイクル**: LMCacheManagerが生成・管理
