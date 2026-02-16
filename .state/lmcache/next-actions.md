# 次にやるべきこと

## 最優先

- [ ] Phase 2: CacheBlend 内部実装深堀り
  - Blenderの計算ロジック（重要token同定、再計算）
  - SegmentTokenDatabaseの動作
  - BlendServerの構造

## 次の優先

- [ ] Phase 2: LookupClient/Server 深堀り（非同期lookup、async_lookup_and_prefetch全体フロー）
- [ ] Phase 2: RemoteBackend リファレンス実装（redis_connector or fs_connector）
  - Serde（CacheGen圧縮含む）のデータ変換パス
- [ ] Phase 2: CacheController / 分散管理
