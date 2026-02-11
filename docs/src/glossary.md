# 用語集

<!-- 調査中に発見した対象OSS固有の用語をここに蓄積する -->

### PagedAttention
KVキャッシュをOSの仮想メモリページングに着想を得て、固定サイズのブロック単位で管理する技術。連続したGPUメモリ確保が不要になり、メモリ断片化を大幅に抑制する。SOSP 2023論文で提案。

**参照**: `target/vllm/csrc/attention/` (カーネル実装)

### Continuous Batching
リクエストの到着・完了に応じてバッチを動的に更新する手法。固定バッチサイズと異なり、GPU稼働率を最大化できる。vLLMのSchedulerが担う。

### Prefill
プロンプト入力トークン全体を処理してKVキャッシュに書き込む最初のフェーズ。計算量が多く、GPUの並列性を活かしやすい。

### Decode
生成済みコンテキストのKVキャッシュを参照しながら次のトークンを1つずつ逐次生成するフェーズ。メモリバウンドになりやすい。

### Chunked Prefill
Prefillフェーズをチャンクに分割してDecodeフェーズと交互実行する手法。長いプロンプトがDecodeのレイテンシを増加させるのを防ぐ。

### EngineCore
vLLMの推論ループの内側コンポーネント。別プロセス（`EngineCoreProc`）として動作し、ZeroMQソケットで上位エンジン層と通信する。Scheduler、KVCacheManager、Executorを統括する。

**参照**: `target/vllm/vllm/v1/engine/core.py:79` (`EngineCore`)

### KVCacheManager
KVキャッシュブロックの割り当て・解放・プレフィックスキャッシュを管理するクラス。BlockPoolを内部で使用する。

**参照**: `target/vllm/vllm/v1/core/kv_cache_manager.py:94` (`KVCacheManager`)

### KVCacheBlock
PagedAttentionで管理するKVキャッシュの最小単位。固定サイズ（block_sizeトークン分）のGPUメモリブロック。

### BlockPool
KVCacheBlockの空きブロックをプール管理するクラス。ブロックの割り当て・返却を効率的に行う。

**参照**: `target/vllm/vllm/v1/core/block_pool.py`

### VllmConfig
全設定を集約するトップレベルクラス。`ModelConfig`、`CacheConfig`、`SchedulerConfig`、`ParallelConfig`等を内包する。

**参照**: `target/vllm/vllm/config/vllm.py`

### GPUModelRunner
GPU上でモデルのフォワードパスを実際に実行するクラス。LoRA、KVConnector、ECConnectorのMixinを持つ。

**参照**: `target/vllm/vllm/v1/worker/gpu_model_runner.py:329` (`GPUModelRunner`)

### Executor
Worker群を管理する抽象層。シングルプロセス（`UniProcExecutor`）、マルチプロセス（`MultiprocExecutor`）、Ray分散（`RayDistributedExecutor`）の実装がある。

**参照**: `target/vllm/vllm/v1/executor/abstract.py`

### Worker
1つのGPU（またはCPU/XPU）デバイスを担当するプロセス。GPUModelRunnerを保持し、Executorから呼び出される。

**参照**: `target/vllm/vllm/v1/worker/gpu_worker.py:70` (`Worker`)

### Speculative Decoding
ドラフトモデル（小さいモデル）で複数トークンを仮生成し、メインモデルで一括検証することで推論を高速化する手法。

**参照**: `target/vllm/vllm/v1/spec_decode/`

### LoRA (Low-Rank Adaptation)
少量の追加パラメータでLLMをファインチューニングする手法。vLLMは複数LoRAの動的切替（Multi-LoRA）をランタイムでサポートする。

**参照**: `target/vllm/vllm/lora/`

### KV Transfer
複数のvLLMインスタンス間でKVキャッシュを転送する機構。Disaggregated Prefill（PrefillとDecodeを異なるインスタンスで実行）等に使用。KVConnector抽象基底クラスとして実装され、LMCache、NIXL、P2P NCCL、Mooncake等の複数バックエンドがある。

**参照**: `target/vllm/vllm/distributed/kv_transfer/` (全体), `target/vllm/vllm/config/kv_transfer.py:17` (`KVTransferConfig`)

### LMCache
vLLMと統合可能な外部KVキャッシュストレージ。KV Transferのバックエンドの一つとして動作し、KVキャッシュのCPUオフロード、インスタンス間共有等を提供する。

**参照**: `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`, `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/`

### Multimodal (マルチモーダル)
テキスト以外の入力（画像・動画・音声）を扱うモデル機能。`vllm/multimodal/` にプロセッサ・レジストリ等が実装されている。Gemma3等のマルチモーダルモデルが対応。

**参照**: `target/vllm/vllm/multimodal/`

### Unified Compute Model
vLLM v1のSchedulerが採用するスケジューリングアプローチ。PrefillフェーズとDecodeフェーズを明示的に区別せず、各リクエストの`num_computed_tokens`（計算済みトークン数）が目標に追いつくまでトークンを割り当てる。これにより、Chunked Prefill、Prefix Caching、Speculative Decodingを統一的に扱える。

**参照**: `target/vllm/vllm/v1/core/sched/scheduler.py:322` (コメント)

### collective_rpc
Executor層が全Workerに対して同一メソッドを実行するRPCパターン。メソッド名（文字列）または関数を受け取り、全Workerで並列実行後、出力ランクのWorkerの結果を返す。`non_block=True`でFuture返却も可能。

**参照**: `target/vllm/vllm/v1/executor/abstract.py:180` (`collective_rpc`)

### ExecuteModelState
GPUModelRunnerの2フェーズ実行パターンで使用される一時状態。execute_model()がlogitsやhidden_statesなどのGPUテンソルを保存し、sample_tokens()が復元してサンプリングを行う。NamedTuple。

**参照**: `target/vllm/vllm/v1/worker/gpu_model_runner.py:313` (`ExecuteModelState`)

### OutputProcessor
フロントエンドプロセスで動作し、EngineCoreOutputをRequestOutputに変換するコンポーネント。インクリメンタルデトークナイズ、停止文字列判定、logprobs処理を行う。

**参照**: `target/vllm/vllm/v1/engine/output_processor.py:73` (`OutputProcessor`)

### IncrementalDetokenizer
トークンIDからテキストへのインクリメンタル変換を行うクラス。FastIncrementalDetokenizer（HF DecodeStream）とSlowIncrementalDetokenizer（Python実装）の2種がある。

**参照**: `target/vllm/vllm/v1/engine/detokenizer.py:30` (`IncrementalDetokenizer`)

### RequestOutputKind
出力モードを定義するEnum。CUMULATIVE（毎回全出力）、DELTA（差分のみ、ストリーミング向け）、FINAL_ONLY（完了時のみ）の3値。

**参照**: `target/vllm/vllm/sampling_params.py:108` (`RequestOutputKind`)

### mm_cache (マルチモーダルキャッシュ)
マルチモーダル入力（画像エンコーダ出力等）のキャッシュ機構。同一画像の繰り返し処理を避けるため、エンコーダ出力をキャッシュする。

**参照**: `target/vllm/vllm/v1/worker/gpu/mm/encoder_runner.py`

### FreeKVCacheBlockQueue
空きKVキャッシュブロックをLRU順序で管理する双方向リンクリスト。Pythonの`deque`ではなく独自実装を採用し、O(1)の中間要素削除をサポートする。センチネルノード（fake_head/fake_tail）を使用。

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:156`

### BlockHashWithGroupId
`BlockHash`（ブロックのハッシュ値）にKVキャッシュグループID（4バイトBE）を結合したバイト列。Tuple生成を避けてGC負荷を低減する。プレフィックスキャッシュのキーとして使用。

**参照**: `target/vllm/vllm/v1/core/kv_cache_utils.py:39`

### null_block
BlockPoolが保持する特殊なKVCacheBlock（block_id=0, is_null=True）。Sliding Window Attentionのウィンドウ外位置やMambaのスキップ位置を埋めるプレースホルダ。物理メモリを消費せず、解放・Eviction対象外。

**参照**: `target/vllm/vllm/v1/core/block_pool.py:174`

### KVCacheCoordinator
複数のKVキャッシュグループ（異なるアテンションタイプのレイヤー群）を統括する抽象クラス。NoPrefixCache、Unitary（単一グループ）、Hybrid（複数グループ）の3実装がある。

**参照**: `target/vllm/vllm/v1/core/kv_cache_coordinator.py:28`

### SingleTypeKVCacheManager
1種類のアテンションタイプのKVキャッシュ管理ロジックを担当する抽象基底クラス。FullAttention、SlidingWindow、ChunkedLocalAttention、Mamba、CrossAttention、SinkFullAttentionの7実装がある。

**参照**: `target/vllm/vllm/v1/core/single_type_kv_cache_manager.py:24`

### Cascade Attention
全リクエストで共有される共通プレフィックスの再計算をスキップする最適化。`get_num_common_prefix_blocks()`で共通ブロック数を判定し、アテンション計算から除外する。

### Sliding Window Attention
各トークンが直近のN個のトークンにのみアテンションするメカニズム。ウィンドウ外のKVキャッシュブロックは`null_block`で置換されメモリを節約する。

### Attention Sink (StreamingLLM)
先頭の少数トークン（sink tokens）のKVキャッシュを常に保持しつつ、中間トークンを捨てて長いシーケンスを処理する手法。`SinkFullAttentionManager`が実装。
