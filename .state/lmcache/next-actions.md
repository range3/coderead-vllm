# 次にやるべきこと

## 最優先

- [ ] Phase 1: 垂直スライス — 「vLLM統合でのKVキャッシュ store/retrieve パス」
  - エントリポイント: LMCacheConnectorV1Dynamic → LMCacheConnectorV1Impl
  - コアパス: LMCacheManager → LMCacheEngine → TokenDatabase → GPUConnector → StorageManager → LocalCPUBackend
  - 目標: 各コンポーネントのインターフェースと接続点を理解し、独自プラグイン作成の基盤知識を得る
  - セッション1: store パス（GPU→CPU→Storage）を追跡
  - セッション2: retrieve パス（Storage→CPU→GPU）を追跡
  - （必要に応じてセッション3で補足）

## 次の優先

- [ ] Phase 2: StorageManager + LocalCPUBackend 深堀り（Disk階層化含む）
- [ ] Phase 2: CacheBlend 内部実装深堀り
