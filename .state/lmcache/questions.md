# 未解決の疑問

## Phase 1 セッション2で解決すべき疑問

- [ ] retrieve 時の prefetch はどう動作するか？StorageManager 内でどう調整される？
- [ ] GPUConnector の to_gpu で使われる slot_mapping はどこから来るか？（store側と同じ経路か？）
- [ ] In-Process モードと MP モードでの LMCacheEngine の生成・利用パスの違いは？
- [ ] LookupClient は Scheduler プロセスで動作するが、LMCacheEngine 本体は Worker プロセス。この間の通信は？
- [ ] retrieve時のMemoryObj取得と解放のライフサイクルは？

## Phase 1で解決済み

- [x] LMCacheEngine の store 処理はどのタイミングで呼ばれるか？
  → vLLMの各attentionレイヤー実行後に`save_kv_layer()`フックから呼ばれる。Layer 0でGenerator生成、以降`next()`で進行
- [x] TokenDatabase の chunk_hash 計算は vLLM 側のプレフィックスキャッシュと完全に同一のアルゴリズムか？
  → **はい、完全に同一**。vLLMの`get_hash_fn_by_name("sha256_cbor")`を直接importして使用。NONE_HASHもvLLMから取得
- [x] GPUConnector の from_gpu で使われる slot_mapping はどこから来るか？
  → ConnectorMetadata.requests[i].slot_mapping（Scheduler側で構築、Worker側に渡される）
- [x] StorageManager の各バックエンド間のデータ移動（L1→L2プロモーション等）はどう管理されるか？
  → `batched_put()`が全バックエンドに同時配布。プロモーションではなく同時書き込み。各バックエンドのallocatorでMemoryObjを別途確保しコピー

## Phase 2以降

- [ ] LocalCPUBackend の詳細実装（メモリ確保戦略、Eviction実装）
- [ ] 独自ストレージバックエンドを作る場合に実装すべきインターフェースは？
- [ ] CacheBlend の Blender がレイヤー単位でどう計算を実行するか？
- [ ] Serde の CacheGen 圧縮率と性能特性は？
