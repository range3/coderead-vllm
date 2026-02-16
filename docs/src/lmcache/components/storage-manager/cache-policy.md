# CachePolicy — Eviction戦略

> **深度**: [DEEP] / **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 2 セッション1）

## 概要

キャッシュのEviction（追い出し）戦略を抽象化するポリシーフレームワーク。
LocalCPUBackendとLocalDiskBackendが独立したCachePolicyインスタンスを持つ。

**参照**: `target/LMCache/lmcache/v1/storage_backend/cache_policy/`

## BaseCachePolicy インターフェース

**参照**: `target/LMCache/lmcache/v1/storage_backend/cache_policy/base_policy.py`

```python
class BaseCachePolicy(Generic[KeyType, MapType]):
    def init_mutable_mapping(self) -> MapType
    def update_on_hit(self, key, cache_dict) -> None
    def update_on_put(self, key) -> None
    def update_on_force_evict(self, key) -> None
    def get_evict_candidates(self, cache_dict, num_candidates=1) -> list[KeyType]
```

**重要な設計判断**:
- `init_mutable_mapping()`がhot_cache/dictの型を決定（dict, OrderedDict等）
- `get_evict_candidates()`は**best effort**: `can_evict`チェックでpinned/参照中のオブジェクトをスキップ
- `cache_dict`の値がMemoryObj（hot_cache）またはDiskCacheMetadata（disk dict）

## FIFO — First In, First Out

**参照**: `target/LMCache/lmcache/v1/storage_backend/cache_policy/fifo.py`

| メソッド | 実装 |
|---------|------|
| `init_mutable_mapping()` | `dict`（Python dictは挿入順保持） |
| `update_on_hit()` | **何もしない**（FIFOはアクセスで順序不変） |
| `update_on_put()` | **何もしない**（dictの末尾に自然追加） |
| `update_on_force_evict()` | **何もしない** |
| `get_evict_candidates()` | dict先頭からイテレート、`can_evict`なものを返す |

最もシンプル。追い出し候補の選定はO(k)（kは先頭のnon-evictableエントリ数）。

## LRU — Least Recently Used

**参照**: `target/LMCache/lmcache/v1/storage_backend/cache_policy/lru.py`

| メソッド | 実装 |
|---------|------|
| `init_mutable_mapping()` | `OrderedDict` |
| `update_on_hit()` | `cache_dict.move_to_end(key)` + chunk再利用時間追跡 |
| `update_on_put()` | chunk初回タイムスタンプ記録 |
| `update_on_force_evict()` | **何もしない** |
| `get_evict_candidates()` | OrderedDict先頭（最も古いアクセス）からイテレート |

**追加機能**: `chunk_hash_to_init_timestamp`でチャンクの再利用間隔をPrometheusメトリクスに報告。
メモリ上限 `max_num_chunk_hash=12,500,000`、超過時は辞書をclear()。

## LFU — Least Frequently Used

**参照**: `target/LMCache/lmcache/v1/storage_backend/cache_policy/lfu.py`

| メソッド | 実装 |
|---------|------|
| `init_mutable_mapping()` | `dict` |
| `update_on_hit()` | freq++。freq_to_keys[old_freq]→freq_to_keys[new_freq]に移動 |
| `update_on_put()` | `key_to_freq[key] = 1`、freq_to_keys[1]に登録 |
| `update_on_force_evict()` | key_to_freq/freq_to_keys両方からkey除去 |
| `get_evict_candidates()` | 最低freq→高freqの順にイテレート、同freq内はFIFO |

**データ構造**:
- `freq_to_keys`: `SortedDict[int, dict[key, None]]` — freq順ソート
- `key_to_freq`: `dict[key, int]` — O(1)でfreq逆引き
- 計算量: update_on_hit = O(log N)（SortedDictの操作）

**注意**: get_evict_candidates()内でkey_to_freq.pop()を直接実行（副作用あり）。

## MRU — Most Recently Used

**参照**: `target/LMCache/lmcache/v1/storage_backend/cache_policy/mru.py`

| メソッド | 実装 |
|---------|------|
| `init_mutable_mapping()` | `OrderedDict` |
| `update_on_hit()` | `cache_dict.move_to_end(key, last=True)` |
| `update_on_put()` | **何もしない** |
| `update_on_force_evict()` | **何もしない** |
| `get_evict_candidates()` | `reversed(cache_dict.items())`でOrderedDict末尾（最新アクセス）から |

LRUの逆。ストリーミング的アクセスパターン（同じチャンクが二度使われない場合）に有効。

## Eviction発動フロー

```mermaid
graph TD
    A[allocate() 失敗] --> B{use_hot?}
    B -->|Yes| C[cache_policy.get_evict_candidates\nhot_cache, num=1]
    C --> D{候補あり?}
    D -->|Yes| E[batched_remove\nevict_keys]
    E --> F[allocate再試行]
    F --> G{成功?}
    G -->|No| C
    D -->|No| H{busy_loop?}
    H -->|Yes| I[0.1秒待機]
    I --> C
    H -->|No| J[None返却]
    G -->|Yes| K[MemoryObj返却]
    B -->|No| H
```

**can_evict条件**: `not is_pinned and ref_count == 1`
- pinned: lookup中のチャンク（retrieve完了まで保護）
- ref_count > 1: 他のバックエンドやGPU転送が参照中

## 設定

`LMCacheEngineConfig.cache_policy`で指定:
```yaml
cache_policy: "lru"  # fifo | lru | lfu | mru
```

`get_cache_policy()`ファクトリ関数で対応するポリシーインスタンスを生成。
