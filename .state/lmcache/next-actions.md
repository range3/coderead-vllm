# 次にやるべきこと

## 最優先

- [ ] Phase 1 セッション2: 垂直スライス — 「vLLM統合でのKVキャッシュ retrieve パス」
  - エントリポイント: LMCacheConnectorV1Impl.load_kv_layer() / start_load_kv()
  - コアパス: LMCacheEngine.retrieve_layer() → StorageManager.batched_get() → LocalCPUBackend → GPUConnector.batched_to_gpu()
  - 目標: retrieve側の各コンポーネントのインターフェースと接続点を理解
  - data-flow.mdにretrieveパスを追記
  - 各コンポーネントsummary.mdのretrieve側を追記

## 次の優先

- [ ] Phase 2: StorageManager + LocalCPUBackend 深堀り（Disk階層化含む）
- [ ] Phase 2: CacheBlend 内部実装深堀り
