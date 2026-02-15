# Phase 0a+0b: LMCache オリエンテーション + ユーザー優先度確認

**日付**: 2026-02-16
**対象OSS**: LMCache
**Phase**: 0a + 0b（小規模OSSのため同セッション実施）

## 実施内容

1. READMEと公式情報確認
2. ディレクトリ構造分析（Python 62K行、C++/CUDA 20ファイル、Rust 1ファイル）
3. 主要エントリポイント6つ特定
4. パッケージ構造ルール8つ策定
5. 13領域のコンポーネント特定
6. ユーザー優先度確認

## 主要発見

### アーキテクチャ
- `lmcache/v1/` が現行アーキテクチャ。トップレベルの `server/`, `storage_backend/` はレガシー
- 2動作モード: In-Process（vLLM内直接）と MultiProcess（ZMQ IPC別プロセス）
- LMCacheEngine（1,949行）がコア。StorageManager（1,145行）がストレージ階層管理
- 15+リモートストレージコネクタ（Redis, S3, Valkey, Mooncake, Infinistore等）

### vLLM統合
- `LMCacheConnectorV1Dynamic` が vLLMの `KVConnectorBase_V1` を実装
- 実装本体は `LMCacheConnectorV1Impl`（vllm_v1_adapter.py）に委譲
- `LMCacheManager` がライフサイクル全体（Engine, LookupClient, OffloadServer等）を管理

### データ構造
- `CacheEngineKey`: (model_name, world_size, worker_id, chunk_hash, dtype) の5タプル
- `MemoryObj`: KVキャッシュデータ + MemoryFormat情報
- `MemoryFormat`: KV_2LTD, KV_T2D, KV_2TD, BINARY, KV_MLA_FMT等
- `LMCacheMetadata`: モデルメタ情報（サービングエンジンから抽出）

### ユーザー優先度
1. コアエンジン + vLLM統合
2. StorageManager + Backends（特にLocalCPU/Disk階層化の詳細実装）
3. CacheBlend
4. 分散/クラスタ管理
- 目的: 独自プラグイン作成準備
- スキップなし（全領域一通り見たい）

## 成果物
- `docs/src/lmcache/architecture/overview.md` — 全体アーキテクチャ [SHALLOW]
- `docs/src/lmcache/glossary.md` — 用語集（30+用語）[VERIFIED]
- `.state/lmcache/reading-guide.md` — 構造ルール8つ + ユーザー優先度

## Phase 1 計画
- 垂直スライス: vLLM統合でのKVキャッシュ store/retrieve パス
- セッション1: store パス、セッション2: retrieve パス
