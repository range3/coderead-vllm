# Phase 2a: KVCacheManager 深堀り

**日付**: 2026-02-11
**フェーズ**: Phase 2（コンポーネント別深堀り）
**目的**: KVCacheManager を [MEDIUM] → [DEEP] に昇格

## 調査範囲

6ファイル、計4,371行を調査:
- `kv_cache_manager.py` (490行) — KVCacheManager、KVCacheBlocks
- `kv_cache_coordinator.py` (586行) — Coordinator 3実装
- `single_type_kv_cache_manager.py` (1065行) — Manager 7種
- `block_pool.py` (490行) — BlockPool、BlockHashToBlockMap
- `kv_cache_utils.py` (1644行) — KVCacheBlock、Queue、ハッシュ計算
- `kv_cache_metrics.py` (96行) — メトリクス収集

## 成果物

### 新規作成（3ファイル）
1. `docs/src/components/kv-cache-manager/block-pool.md` [DEEP]
   - KVCacheBlock、FreeKVCacheBlockQueue、BlockHashToBlockMap、null_block
   - Eviction メカニズム、KV Cache Events、メトリクス収集
2. `docs/src/components/kv-cache-manager/prefix-cache.md` [DEEP]
   - ハッシュチェーン計算、NONE_HASH、BlockHash型階層
   - Extra Keys（MM/LoRA/salt/embeds）、4種Lookupアルゴリズム
   - Hybrid fixed-point、BlockHashListWithBlockSize
3. `docs/src/components/kv-cache-manager/attention-type-managers.md` [DEEP]
   - 基底クラス詳細、7種Manager実装
   - 比較表（スキップ計算、キャッシュ検索、Cascade、DCP/PCP、EAGLE対応）

### 更新（2ファイル）
4. `docs/src/components/kv-cache-manager/summary.md` [MEDIUM] → [DEEP]
   - Coordinator選択ロジック、KVCacheBlocks、KVキャッシュグループ概念
   - allocate_slots() 5段階フロー、3サブドキュメントへのリンク
5. `docs/src/glossary.md` — 8用語追加

## 主要な発見

### BlockPool
- FreeKVCacheBlockQueue は独自双方向リンクリスト（O(1) 中間削除、GC軽量化）
- BlockHashToBlockMap は Union型（KVCacheBlock | dict）でGCコスト削減、重複排除なし設計
- null_block（block_id=0）は解放・Eviction対象外の特殊ブロック
- メトリクス収集はサンプリングベース（デフォルト1%）

### プレフィックスキャッシュ
- ハッシュチェーン: hash(parent_hash, tokens, extra_keys) で各ブロックが前ブロックに依存
- NONE_HASH: ランダム32バイト or PYTHONHASHSEED から決定論的生成
- 4種ハッシュ関数: sha256/sha256_cbor（デフォルト）/xxhash/xxhash_cbor
- Extra Keys: LoRA名、MM identifier、cache_salt（先頭ブロックのみ）、prompt_embeds
- BlockHashListWithBlockSize: Hybrid model 向けのハッシュ粒度遅延変換

### アテンションタイプ別Manager
- 7種: Full、SlidingWindow、ChunkedLocal、Mamba、Cross、SinkFull + 基底クラス
- spec_manager_map でKVCacheSpec型→Managerクラスをディスパッチ
- MambaManager が最も複雑（align/none 2モード、状態ブロック追跡、speculative blocks再利用）
- SinkFullAttentionManager は初期化時にsinkブロックを事前確保
- HybridKVCacheCoordinator の find_longest_cache_hit は反復固定点アルゴリズム

## 解決した疑問
- block_size の設定方法: KVCacheSpecからモデル依存、DCP/PCP倍率あり
- プリエンプション閾値: 明示的閾値なし、allocate_slots()の空きブロック比較で動的判定

## 新たな疑問
- HybridKVCacheCoordinatorの反復固定点は実際のモデルで何回イテレーションするか？
- BlockHashToBlockMapのUnion型最適化の実測パフォーマンス差は？
