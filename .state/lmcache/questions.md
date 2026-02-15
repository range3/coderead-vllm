# 未解決の疑問

## Phase 1で解決すべき疑問

- [ ] LMCacheEngine の store 処理はどのタイミングで呼ばれるか？（vLLM側のどのフックから？）
- [ ] retrieve 時の prefetch はどう動作するか？StorageManager 内でどう調整される？
- [ ] TokenDatabase の chunk_hash 計算は vLLM 側のプレフィックスキャッシュと完全に同一のアルゴリズムか？
- [ ] GPUConnector の to_gpu/from_gpu で使われる slot_mapping はどこから来るか？
- [ ] StorageManager の各バックエンド間のデータ移動（L1→L2プロモーション等）はどう管理されるか？
- [ ] In-Process モードと MP モードでの LMCacheEngine の生成・利用パスの違いは？
- [ ] LookupClient は Scheduler プロセスで動作するが、LMCacheEngine 本体は Worker プロセス。この間の通信は？

## Phase 2以降

- [ ] LocalCPUBackend の詳細実装（メモリ確保戦略、Eviction実装）
- [ ] 独自ストレージバックエンドを作る場合に実装すべきインターフェースは？
- [ ] CacheBlend の Blender がレイヤー単位でどう計算を実行するか？
- [ ] Serde の CacheGen 圧縮率と性能特性は？
