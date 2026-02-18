# 探索ログ

## 現在のフェーズ

Phase 2 進行中（セッション2完了）

## カバレッジマップ

| 領域 | 深度 | 最終更新 | 関連ドキュメント |
|------|------|---------|----------------|
| 全体アーキテクチャ | [SHALLOW] | 2026-02-16 | architecture/overview.md |
| データフロー（store+retrieveパス） | [MEDIUM] | 2026-02-16 | architecture/data-flow.md |
| LMCacheEngine | [MEDIUM] | 2026-02-16 | components/cache-engine/summary.md |
| StorageManager + Backends | [DEEP] | 2026-02-16 | components/storage-manager/summary.md |
| GPUConnector | [MEDIUM] | 2026-02-16 | components/gpu-connector/summary.md |
| TokenDatabase | [MEDIUM] | 2026-02-16 | components/token-database/summary.md |
| vLLM統合(Integration) | [MEDIUM] | 2026-02-16 | components/vllm-integration/summary.md |
| LookupClient/Server | [MEDIUM] | 2026-02-16 | components/vllm-integration/summary.md, data-flow.md |
| MemoryAllocator階層 | [DEEP] | 2026-02-16 | components/storage-manager/memory-allocator.md |
| CachePolicy(Eviction) | [DEEP] | 2026-02-16 | components/storage-manager/cache-policy.md |
| LocalDiskBackend | [DEEP] | 2026-02-16 | components/storage-manager/local-disk-backend.md |
| MemoryManagement | [SHALLOW] | 2026-02-16 | overview.md内 |
| CacheBlend | [MEDIUM] | 2026-02-19 | investigations/cacheblend.md |
| MultiProcess Server | [SHALLOW] | 2026-02-16 | overview.md内 |
| CacheController | [SHALLOW] | 2026-02-16 | overview.md内 |
| Disaggregated Prefill | [SHALLOW] | 2026-02-16 | overview.md内 |
| Serde/圧縮 | [SHALLOW] | 2026-02-16 | overview.md内 |

## セッション履歴

| 日付 | Phase | 概要 | 詳細 |
|------|-------|------|------|
| 2026-02-16 | 0a+0b | オリエンテーション + ユーザー優先度確認 | sessions/20260216-phase0a-orientation.md |
| 2026-02-16 | 1-s1 | Store パス垂直スライス | sessions/20260216-phase1-store-path.md |
| 2026-02-16 | 1-s2 | Retrieve パス垂直スライス | sessions/20260216-phase1-retrieve-path.md |
| 2026-02-16 | 2-s1 | StorageManager+LocalCPUBackend DEEP化 | sessions/20260216-phase2-storage-deep.md |
| 2026-02-19 | 2-s2 | CacheBlend MEDIUM化（vLLM連携方法含む） | sessions/20260219-phase2-cacheblend.md |
