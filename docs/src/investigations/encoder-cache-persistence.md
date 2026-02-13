# EncoderCache 永続化と階層キャッシュ: 調査報告

> **ステータス**: 調査完了 — ECConnector 既存インフラの発見により設計方針が確定
> **作成日**: 2026-02-14
> **深度**: [MEDIUM]
> **確信度**: [VERIFIED]
> **関連ドキュメント**: [Gemma3 ビジョンパイプライン: キャッシュ機構](./gemma3-vision-caches.md)、[マルチモーダル バックエンド MM 処理](../components/multimodal/mm-engine-gpu.md)

---

## 1. 背景と動機

### 現状の EncoderCache

vLLM の EncoderCache は、ビジョンエンコーダ（例: SiglipVisionModel + Projector）の GPU 上の出力テンソルをキャッシュする。Gemma3 27B の場合、出力形状は `(N×256, 5376)` で、1 画像あたり約 2.6 MB（FP16）。

| 項目 | 現状 |
|------|------|
| 格納先 | GPU メモリ（`gpu_model_runner.encoder_cache`） |
| キャッシュキー | `mm_hash` or `{lora_name}:{mm_hash}` |
| Eviction 方式 | **FIFO**（`OrderedDict.popitem(last=False)`） |
| 容量設定 | `encoder_cache_size`（エンベディング数単位） |
| 永続性 | なし（プロセス終了で消失） |
| 管理 | `EncoderCacheManager`（CPU 側論理管理）+ `encoder_cache` dict（GPU 側物理格納） |

**参照**: `target/vllm/vllm/v1/core/encoder_cache_manager.py` (EncoderCacheManager)、`target/vllm/vllm/v1/worker/gpu_model_runner.py:439` (encoder_cache dict)

### RAG ユースケースにおける課題

RAG では同一ドキュメント画像が異なるクエリから繰り返し参照される。

1. **FIFO Eviction との相性の悪さ**: 高頻度アクセス画像でも新しいエントリが来れば古い順に追い出される
2. **GPU メモリの有限性**: `encoder_cache_size` を大きくしてもデコーダ KV Cache と競合
3. **再起動耐性がない**: vLLM プロセス再起動で全キャッシュが消失

### エンコーダ処理のコスト

EncoderCache がスキップする GPU 上の処理:
- SiglipVisionModel: Conv2d + position_embedding + **27 層 Transformer Encoder**（双方向 Attention）+ post_layernorm
- Gemma3MultiModalProjector: AvgPool2d + GemmaRMSNorm + Linear(1152→5376)
- split + flatten

RAG コーパスが数千〜数万画像規模の場合、毎回の再計算コストは無視できない。

---

## 2. 重要な発見: ECConnector 既存インフラ

### KV Transfer ではなく ECConnector が正解

当初の仮説では「KV Transfer の枠組み（LMCache 等）を活用」する方針だったが、調査の結果、**エンコーダキャッシュの外部ストレージ永続化のために設計された専用インフラ「ECConnector」が既に存在**することが判明した。

### ECConnector の全体像

```
target/vllm/vllm/distributed/ec_transfer/
  __init__.py                      -- get_ec_transfer(), has_ec_transfer()
  ec_transfer_state.py             -- グローバルシングルトン管理
  ec_connector/
    __init__.py
    base.py                        -- ECConnectorBase (抽象基底クラス)
    factory.py                     -- ECConnectorFactory (プラグイン登録)
    example_connector.py           -- ECExampleConnector (参照実装, 199行)
```

**参照**: `target/vllm/vllm/distributed/ec_transfer/ec_connector/base.py:59` (ECConnectorBase)

### ECConnectorBase の抽象メソッド [VERIFIED]

`ECConnectorBase` は `KVConnectorBase_V1` とは**完全に独立**した抽象基底クラス。

| メソッド | 分類 | 説明 |
|----------|------|------|
| `start_load_caches(encoder_cache, **kwargs)` | Worker | 外部ストレージから `encoder_cache` dict にテンソルをロード |
| `save_caches(encoder_cache, mm_hash, **kwargs)` | Worker | `encoder_cache` から外部ストレージにテンソルを保存 |
| `has_cache_item(identifier)` | Scheduler | 外部ストレージにキャッシュが存在するか確認 |
| `update_state_after_alloc(request, index)` | Scheduler | アロケーション後の内部状態更新 |
| `build_connector_meta(scheduler_output)` | Scheduler | Scheduler → Worker 間のメタデータ構築 |

**参照**: `target/vllm/vllm/distributed/ec_transfer/ec_connector/base.py:126-224`

### ECConnector の統合ポイント [VERIFIED]

既にGPUModelRunnerとSchedulerに統合済み:

**Scheduler 側** (`target/vllm/vllm/v1/core/sched/scheduler.py:1197-1203`):
```python
if self.ec_connector is not None and self.ec_connector.has_cache_item(
    item_identifier
):
    mm_hashes_to_schedule.add(item_identifier)
    external_load_encoder_input.append(i)
    num_embeds_to_schedule += num_encoder_embeds
    continue
```

→ ECConnector にキャッシュが存在する場合、エンコーダ計算バジェットを消費せず、ロード予約のみ行う。

**Worker 側** (`target/vllm/vllm/v1/worker/gpu_model_runner.py:2444-2445`):
```python
self.encoder_cache[mm_hash] = output
self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

→ エンコーダ実行後、結果を GPU dict に格納すると同時に ECConnector にも保存。

**Worker 側コンテキストマネージャ** (`target/vllm/vllm/v1/worker/ec_connector_model_runner_mixin.py:62-85`):
```python
ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)
if not ec_connector.is_producer:
    ec_connector.start_load_caches(encoder_cache, **kwargs)
try:
    yield output   # _execute_mm_encoder() + _gather_mm_embeddings() が実行される
finally:
    output.finished_sending, output.finished_recving = (
        ec_connector.get_finished(scheduler_output.finished_req_ids)
    )
    ec_connector.clear_connector_metadata()
```

→ `start_load_caches()` でストレージからロード → エンコーダ実行（ロード済みはスキップ）→ 完了通知。

### ECTransferConfig [VERIFIED]

**参照**: `target/vllm/vllm/config/ec_transfer.py`

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `ec_connector` | `str \| None` | None | コネクタ名（例: `"ECExampleConnector"`） |
| `ec_role` | `ECRole \| None` | None | `"ec_producer"` or `"ec_consumer"` |
| `ec_connector_extra_config` | `dict` | `{}` | コネクタ固有の追加設定 |
| `ec_connector_module_path` | `str \| None` | None | 動的ロード用モジュールパス |
| `engine_id` | `str \| None` | uuid4 自動生成 | エンジンID |

起動時パラメータ例: `--ec-connector ECExampleConnector --ec-role ec_producer --ec-connector-extra-config '{"shared_storage_path": "/mnt/cache"}'`

### ECConnectorFactory [VERIFIED]

**参照**: `target/vllm/vllm/distributed/ec_transfer/ec_connector/factory.py`

- `register_connector(name, module_path, class_name)` で遅延ロード登録
- `ec_connector_module_path` による動的ロード（外部モジュール対応）
- 現在の登録済みコネクタ: `ECExampleConnector` のみ

---

## 3. ECExampleConnector 参照実装の分析 [VERIFIED]

**参照**: `target/vllm/vllm/distributed/ec_transfer/ec_connector/example_connector.py`

### 概要

safetensors を使ってディスクにエンコーダ出力テンソルを保存/読込するデバッグ用実装。全 199 行。

### ストレージ構造

```
{shared_storage_path}/
  {mm_hash}/
    encoder_cache.safetensors
```

### 保存 (save_caches, L98-118)

```python
def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
    if not self.is_producer:
        return
    filename = self._generate_filename_debug(mm_hash)
    ec_cache = encoder_cache[mm_hash]
    tensors = {"ec_cache": ec_cache.detach().cpu()}  # GPU→CPU コピー
    safetensors.torch.save_file(tensors, filename)
```

- Producer ロールの場合のみ保存
- `detach().cpu()` で GPU テンソルを CPU に移動してからシリアライズ
- **テンソル形状に一切依存しない**

### ロード (start_load_caches, L63-96)

```python
def start_load_caches(self, encoder_cache, **kwargs) -> None:
    metadata = self._get_connector_metadata()
    for mm_data in metadata.mm_datas:
        if mm_data.mm_hash in encoder_cache:
            continue  # 既に GPU dict にあればスキップ
        filename = self._generate_filename_debug(mm_data.mm_hash)
        ec_cache = safetensors.torch.load_file(filename, device=...)["ec_cache"]
        encoder_cache[mm_data.mm_hash] = ec_cache  # dict に直接格納
```

- メタデータ（Scheduler が構築）に基づいてロード対象を決定
- `encoder_cache` dict に直接格納 → `_gather_mm_embeddings()` でそのまま読める

### 存在チェック (has_cache_item, L120-133)

```python
def has_cache_item(self, identifier: str) -> bool:
    return self._found_match_for_mm_data(identifier)
    # → os.path.exists(filename)
```

- ファイルの存在確認のみ（同期的）

### メタデータ管理

`ECExampleConnectorMetadata` (L35-42): ロードすべき `mm_hash` と `num_token` のリスト。

`update_state_after_alloc()` (L135-146): Scheduler が allocate() 後に呼び出し、`_mm_datas_need_loads` にロード対象を追加。

`build_connector_meta()` (L148-164): `_mm_datas_need_loads` からメタデータを構築し、Worker に伝達。呼び出し後にクリア。

---

## 4. KV Transfer との比較 [VERIFIED]

| 評価項目 | KV Transfer | ECConnector |
|---|---|---|
| **設計目的** | デコーダ KV Cache の転送・永続化 | エンコーダ出力テンソルの転送・永続化 |
| **テンソル粒度** | レイヤー別、ブロック単位、トークン粒度 | `mm_hash` 単位、任意形状テンソル |
| **テンソル形状依存** | あり (`num_layer, 2, chunk_size, num_kv_heads, head_size`) | **なし** |
| **エンコーダ出力への適合性** | 不適合 | **最適** |
| **既存統合ポイント** | Attention 層デコレータ経由 | GPUModelRunner の `_execute_mm_encoder` 直後 |
| **新規実装量** | 大（7 abstract メソッド + KV 概念適合） | **小**（5 abstract メソッド、参照実装 199 行） |
| **ストレージ実装** | LMCache/NIXL/Mooncake（全て KV 前提） | Example（safetensors/ディスク）、拡張容易 |

**結論**: エンコーダ出力テンソルの永続化には ECConnector を使うべき。KV Transfer はデコーダ KV Cache に特化しており、エンコーダ出力の形状・粒度に合わない。

LMCache の KV 形状ハードコード箇所:
- `target/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:477`
  ```python
  kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_heads, head_size)
  ```

---

## 5. FIFO → LRU 変更の具体的設計

### 現状の FIFO 実装 [VERIFIED]

**参照**: `target/vllm/vllm/v1/core/encoder_cache_manager.py`

**データ構造**:
```python
# L72-77
self.cached: dict[str, set[str]] = {}           # mm_hash → {request_ids}
self.freeable: OrderedDict[str, int] = OrderedDict()  # mm_hash → num_embeds (挿入順)
self.freed: list[str] = []                       # evict 済みリスト
```

**FIFO の核心** (L173-177):
```python
while num_embeds > self.num_free_slots:
    mm_hash, num_free_embeds = self.freeable.popitem(last=False)  # 最も古いエントリから
    del self.cached[mm_hash]
    self.freed.append(mm_hash)
    self.num_free_slots += num_free_embeds
```

`OrderedDict.popitem(last=False)` で**最初に挿入された（＝最も早く参照解放された）エントリ**から Evict。

### 現状の FIFO が LRU と異なる点

FIFO は「最も早く `freeable` に追加されたものから Evict」する。LRU は「最も長期間アクセスされていないものから Evict」する。

差が出るケース:
1. 画像 A が freeable に入る（参照解放）
2. 画像 B が freeable に入る
3. 画像 A が再度参照される → freeable から取り出されて active に戻る
4. 画像 A が再度 freeable に入る → **FIFO: 末尾に追加（最新扱い）、LRU: 末尾に追加（最新扱い）**

実は、**現状の FIFO は「参照解放順」であり、再参照された画像は freeable の末尾に再挿入される**ため、RAG での繰り返しアクセスパターンでは擬似 LRU として機能する部分もある。

しかし、**active 状態（参照中）のエントリ間でのアクセス頻度は考慮されない**。複数リクエストが同時に異なる画像を参照し、それらが一斉に freeable になった場合、「最後にアクセスされた時刻」ではなく「最後に参照解放された時刻」で順序が決まる。

### LRU への変更方法

**変更箇所**: `encoder_cache_manager.py` の 1 ファイルのみ。Scheduler 側・GPUModelRunner 側の変更は不要。

**方法 A: アクセス時刻の追跡**（推奨）

```python
class EncoderCacheManager:
    def __init__(self, cache_size: int):
        # ... 既存フィールド ...
        self._access_order: dict[str, int] = {}  # mm_hash → monotonic counter
        self._access_counter: int = 0

    def check_and_update_cache(self, request, input_id) -> bool:
        mm_hash = request.mm_features[input_id].identifier
        if mm_hash not in self.cached:
            return False
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds
        self.cached[mm_hash].add(request.request_id)
        # ★ アクセス時刻を更新
        self._access_counter += 1
        self._access_order[mm_hash] = self._access_counter
        return True

    def free_encoder_input(self, request, input_id) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds
            # ★ アクセス時刻でソートされた位置に挿入
            # OrderedDictを再ソート: 古いアクセスが先頭に来るようにする
            self.freeable.move_to_end(mm_hash)  # 末尾に追加（最新アクセス）

    def allocate(self, request, input_id) -> None:
        mm_hash = request.mm_features[input_id].identifier
        # ... 既存ロジック ...
        # ★ アクセス時刻を記録
        self._access_counter += 1
        self._access_order[mm_hash] = self._access_counter
```

**方法 B: 簡易 LRU（move_to_end のみ）**

現状の実装でも、`freeable` への再挿入は末尾に行われるため、ほぼ LRU として機能する。唯一の改善点は、`check_and_update_cache()` で freeable から復活する際のタイムスタンプ更新のみ。実質的に方法 A と同等の効果が得られる。

### 変更の影響範囲

| コンポーネント | 変更 |
|---|---|
| `encoder_cache_manager.py` | `check_and_update_cache()` と `free_encoder_input()` の 2 メソッド修正 |
| `can_allocate()` | 変更不要（`popitem(last=False)` は同じ） |
| Scheduler | 変更不要（API は同じ） |
| GPUModelRunner | 変更不要 |

---

## 6. 階層キャッシュの実装設計

### アーキテクチャ

```
リクエスト到着
    │
    ▼
Scheduler: _try_schedule_encoder_inputs()
    │
    ├── check_and_update_cache() → L1 HIT (GPU dict) → スキップ
    │
    ├── L1 MISS → ec_connector.has_cache_item() → L2 HIT (Storage)
    │       │
    │       └── external_load_encoder_input に追加 → Worker でロード予約
    │
    └── L1/L2 MISS → encoder_inputs_to_schedule に追加 → エンコーダ計算
    │
    ▼
Worker: execute_model()
    │
    ├── start_load_caches() → L2 からテンソルを GPU dict にロード
    │
    ├── _execute_mm_encoder() → L1/L2 MISS 分のみエンコーダ実行
    │       └── save_caches() → 新規計算結果を L2 に保存
    │
    └── _gather_mm_embeddings() → GPU dict からテンソル取得
```

### 2 層キャッシュの役割分担

| | L1: GPU dict（ホット） | L2: ECConnector（コールド） |
|---|---|---|
| 格納先 | GPU メモリ | Redis / ディスク / NFS 等 |
| 容量 | 小（`encoder_cache_size`） | 大（コーパス全体） |
| レイテンシ | ナノ秒 | マイクロ〜ミリ秒 |
| Eviction | **LRU**（提案変更後） | TTL or LRU or なし |
| 永続性 | なし | あり |
| 管理 | `EncoderCacheManager` | カスタム `ECConnectorBase` 実装 |

### カスタム ECConnector の実装ガイド

新しい ECConnector を作成するには、`ECConnectorBase` を継承して 5 つの abstract メソッドを実装する。

```python
# my_ec_connector.py
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase, ECConnectorMetadata, ECConnectorRole
)

class RedisECConnector(ECConnectorBase):
    def __init__(self, vllm_config, role):
        super().__init__(vllm_config=vllm_config, role=role)
        self._redis_url = vllm_config.ec_transfer_config.get_from_extra_config(
            "redis_url", "redis://localhost:6379"
        )
        # Redis クライアント初期化...

    # Worker側: ストレージからGPU dictにロード
    def start_load_caches(self, encoder_cache, **kwargs):
        metadata = self._get_connector_metadata()
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            tensor_bytes = self._redis.get(mm_data.mm_hash)
            if tensor_bytes:
                encoder_cache[mm_data.mm_hash] = deserialize(tensor_bytes)

    # Worker側: GPU dictからストレージに保存
    def save_caches(self, encoder_cache, mm_hash, **kwargs):
        if not self.is_producer:
            return
        tensor = encoder_cache[mm_hash].detach().cpu()
        self._redis.set(mm_hash, serialize(tensor))

    # Scheduler側: ストレージにキャッシュが存在するか
    def has_cache_item(self, identifier):
        return self._redis.exists(identifier)

    # Scheduler側: アロケーション後の状態更新
    def update_state_after_alloc(self, request, index):
        mm_hash = request.mm_features[index].identifier
        num_token = request.get_num_encoder_embeds(index)
        self._need_loads[mm_hash] = num_token

    # Scheduler側: メタデータ構築
    def build_connector_meta(self, scheduler_output):
        meta = MyECConnectorMetadata()
        for mm_hash, num_token in self._need_loads.items():
            meta.add(mm_hash, num_token)
        self._need_loads.clear()
        return meta
```

**登録方法**:
1. ファクトリ登録: `ECConnectorFactory.register_connector("RedisECConnector", "my.module", "RedisECConnector")`
2. または動的ロード: `--ec-connector RedisECConnector --ec-connector-module-path my.module`

### テンソルサイズの見積もり

Gemma3 27B、1 画像あたり:
```
256 tokens × 5376 dim × 2 bytes (FP16) = 2,752,512 bytes ≈ 2.6 MB/画像
```

| コーパス規模 | L2 ストレージ必要量（FP16） |
|-------------|----------------------|
| 1,000 画像 | ≈ 2.6 GB |
| 10,000 画像 | ≈ 26 GB |
| 100,000 画像 | ≈ 260 GB |

### プリコンピュート運用

ECConnector を活用したオフラインプリコンピュートの流れ:

1. **Producer モード**で vLLM を起動し、コーパス全画像を含むダミーリクエストを送信
2. `save_caches()` でエンコーダ出力がストレージに蓄積される
3. **Consumer モード**で本番 vLLM を起動
4. リクエスト到着時に `has_cache_item()` → `start_load_caches()` でストレージからロード
5. エンコーダ計算をスキップし、ストレージからの読み出し + GPU 転送のみで処理

---

## 7. 残る設計上の考慮事項

### 7.1 ECExampleConnector の同期 I/O

現在の `ECExampleConnector` の `start_load_caches()` は同期的な `safetensors.torch.load_file()` を呼ぶ。ディスク I/O がブロッキングとなり、エンコーダ実行前のレイテンシに直接影響する。

**対策案**:
- `start_load_caches()` を非同期化（別スレッドでロード開始、`_gather_mm_embeddings()` 前に完了待ち）
- Redis 等のインメモリストレージを使い、I/O レイテンシを最小化
- EngineCore.step() のスケジューリングとモデル実行の間の時間的ギャップを活用

### 7.2 LRU とストレージ Eviction の相互作用

L1（GPU dict）から LRU で Evict されたテンソルは、L2（ストレージ）には残る。次にアクセスされた時:
1. Scheduler: `check_and_update_cache()` → L1 MISS
2. Scheduler: `ec_connector.has_cache_item()` → L2 HIT
3. Worker: `start_load_caches()` → L2 から L1 にロード

→ エンコーダ再計算は不要だが、ストレージ→GPU 転送のレイテンシが発生する。

### 7.3 Producer/Consumer ロールの運用

ECConnector は P/D 分離を想定した設計。RAG ユースケースでは:
- **ec_producer**: プリコンピュート用インスタンス（エンコーダ出力をストレージに書き込み）
- **ec_consumer**: 本番サービング用インスタンス（ストレージからロード）
- Producer と Consumer で同じストレージパスを共有する必要がある

### 7.4 キャッシュ無効化

モデル重み更新（LoRA ホットスワップ等）時:
- L1: `EncoderCacheManager.reset()` + `encoder_cache.clear()` で対応済み
- L2: ストレージ側のキャッシュクリアが必要（`identifier` に LoRA プレフィックスが含まれるため、LoRA 別に無効化可能）

---

## 8. 次のステップ

1. **FIFO→LRU の実装**: `encoder_cache_manager.py` の 2 メソッドを修正（変更量: 数行）
2. **カスタム ECConnector の実装**: Redis バックエンドの ECConnector を作成（参照: ECExampleConnector の 199 行）
3. **ベンチマーク**: RAG ワークロードでの比較
   - ベースライン: FIFO + インメモリのみ
   - 改善 1: LRU + インメモリのみ
   - 改善 2: LRU + Redis ECConnector
4. **コミュニティ調査**: vLLM の Issue/PR で ECConnector 関連の議論を確認
