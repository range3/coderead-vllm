# 次にやるべきこと

## 最優先

- [ ] Phase 2: LookupClient/Server 深堀り（非同期lookup、async_lookup_and_prefetch全体フロー）

## 次の優先

- [ ] Phase 2: RemoteBackend リファレンス実装（redis_connector or fs_connector）
  - Serde（CacheGen圧縮含む）のデータ変換パス
- [ ] Phase 2: CacheController / 分散管理
- [ ] Phase 2: CacheBlend DEEP化（TODOが多く将来実装が追加される可能性あり）
  - TP対応状況のフォロー
  - プレフィックスキャッシュとの互換実装

