# LMCache 読解ガイド

> **確信度**: [VERIFIED]
> **最終更新**: 2026-02-16（Phase 0a）

## コードベース構造ルール

### ルール1: v1がメイン、レガシーはスキップ
- `lmcache/v1/` が現行アーキテクチャ。調査対象はここ
- `lmcache/server/`, `lmcache/storage_backend/`（トップレベル）はレガシー→**スキップ**
- `lmcache/config.py`（トップレベル）もレガシー→`lmcache/v1/config.py`を読む

### ルール2: ストレージコネクタは代表実装パターン
- `lmcache/v1/storage_backend/connector/` に15+リモートコネクタ
- **リファレンス実装**: `redis_connector.py` または `fs_connector.py` を深堀り
- 他コネクタ（s3, valkey, mooncake, infinistore等）は差分のみ記録

### ルール3: GPUConnectorも代表実装パターン
- `VLLMPagedMemGPUConnectorV2`がメイン実装（vLLM統合）
- Layerwise系、XPU系は差分記録

### ルール4: observability.pyは大きいがスキップ可
- 1,839行だがPrometheusメトリクス定義が大部分
- 機能理解には不要。メトリクス名を確認する程度でよい

### ルール5: 2つの動作モードを意識
- **In-Process**: vLLM内で直接動作。`integration/vllm/` → `v1/cache_engine.py` が中心
- **MultiProcess (MP)**: 別プロセスサーバー。`v1/multiprocess/` + `v1/distributed/`
- vLLM統合のIn-Processモードを先に理解すべき

### ルール6: usage_context.pyはスキップ
- テレメトリ/利用状況トラッキング。コア機能理解には不要

### ルール7: tools/, benchmarks/, examples/ はスキップ
- ベンチマーク・サンプルコード。アーキテクチャ理解には不要

### ルール8: vLLM統合のバージョン互換コードに注意
- `lmcache_connector_v1_085.py` は vLLM 0.8.5 互換版→スキップ
- `lmcache_connector_v1.py`（latest版）を読む

## ユーザー優先度

### 調査目的
- **独自プラグイン作成準備**: 独自のストレージバックエンドやGPUコネクタを自分で実装できるレベルの知識獲得

### 優先度順
1. **コアエンジン + vLLM統合**: LMCacheEngine全体フロー、vLLM統合のstore/retrieveパス
2. **StorageManager + Backends**: CPU/Disk/Remoteの階層管理、特にLocalCPUBackendの詳細実装とDiskとの階層化
3. **CacheBlend**: 非プレフィックスKVキャッシュ再利用の内部実装
4. **分散/クラスタ管理**: CacheController, P2P, MultiProcess, Disaggregated Prefill

### スキップ対象
- 特になし（全領域一通り見たい）
- ただしルール2/3（代表実装パターン）、ルール4/6/7（observability/usage_context/tools等）のスキップは維持

### Phase 1 垂直スライス提案
- **推奨スライス**: 「vLLM統合でのKVキャッシュ store パス」
  - LMCacheConnectorV1Dynamic → LMCacheConnectorV1Impl → LMCacheManager → LMCacheEngine → TokenDatabase → GPUConnector → StorageManager → LocalCPUBackend
  - これにより、独自プラグイン作成に必要なインターフェース理解が最も効率的に得られる
