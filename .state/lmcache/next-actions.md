# 次にやるべきこと

## 最優先

- [ ] Phase 2: StorageManager + LocalCPUBackend 深堀り（[MEDIUM]→[DEEP]）
  - Eviction実装の詳細（CachePolicy各戦略のコードパス）
  - MemoryAllocatorのプール管理
  - LocalDiskBackendとの階層化動作
  - 独自ストレージバックエンド実装に必要なインターフェース整理

## 次の優先

- [ ] Phase 2: CacheBlend 内部実装深堀り
  - Blenderの計算ロジック（重要token同定、再計算）
  - SegmentTokenDatabaseの動作
  - BlendServerの構造
- [ ] Phase 2: LookupClient/Server 深堀り（非同期lookup、async_lookup_and_prefetch全体フロー）
