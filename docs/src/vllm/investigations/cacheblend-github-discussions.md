# CacheBlend GitHub 議論調査

**調査日**: 2026-02-14
**対象リポジトリ**: vllm-project/vllm, LMCache/LMCache, LMCache/LMCache-Ascend

## エグゼクティブサマリー

CacheBlendは、プレフィックス一致に限定せず、部分的・非連続的なKVキャッシュの再利用を可能にする技術（[論文: arXiv:2405.16444](https://arxiv.org/abs/2405.16444)）。LMCacheプロジェクトがvLLM向けの実装を提供しているが、**vLLM本体への統合はまだ実現していない**。

最大の未解決課題は **オンライン推論（`vllm serve`）でのCacheBlend利用**であり、2026年2月時点でも安定した動作は実現されていない。オフライン（`LLM.generate()`直接呼び出し）では動作するが、HTTP APIを通じた利用では複数の技術的障壁がある。

## 主要な議論・論点

### 1. vLLM本体への Generalized KV Cache Reuse RFC

- **Issue**: [vllm-project/vllm#25950](https://github.com/vllm-project/vllm/issues/25950) (open)
- **状態**: 停滞中（staleラベル後にunstaleされたが、実装は未着手）

プレフィックス一致に限定しない一般化されたKVキャッシュ再利用をvLLMに組み込む提案。3領域の変更を提案:

1. **Attention Kernel**: per-tokenマスキングパラメータの追加（FlashInferは既にサポート）
2. **KV Connector**: 成功ビットマップの返却（現在のsequentialなprefix長ではなく）
3. **Scheduler**: 穴あきKVキャッシュを持つトークンの適切な処理

**2026年1月12日の重要な進展**: カーネルやグルーロジックを変更せずに実装する方法を発見。**プロンプトを複数のサブリクエストに分割してバッチ内で処理する**アプローチ。コードは「soon」とされたが、まだ公開されていない。

LMCacheチーム（@ApostaC）は「CacheBlendは既にLMCacheに実装済み」と返答。提案者（@iddo10）は「LMCacheの実装は別のforwardパスを書いているが、この提案はvLLM本体に組み込む」と差別化を主張。

### 2. `vllm serve` でのCacheBlendサポート（オンライン推論） [最重要]

- **Issue**: [LMCache#1936](https://github.com/LMCache/LMCache/issues/1936) (open)
- **関連**: [#1136](https://github.com/LMCache/LMCache/issues/1136), [#1290](https://github.com/LMCache/LMCache/issues/1290), [#1682](https://github.com/LMCache/LMCache/issues/1682) (いずれもclosed/stale)
- **状態**: 未解決。繰り返し報告されており、最も重要な未解決課題

CacheBlendは現在 `LLM.generate()` によるオフライン推論でのみ動作。`vllm serve` 経由のHTTP APIでは以下の技術的障壁がある:

1. **トークン化の不一致**: `blend_special_str`（例: `" # #"`）が前後のコンテキストにより異なるトークンIDに変換される。HTTP API経由ではトークン化を制御不可能
2. **`/v1/chat/completions` は `input_ids` を受け付けない**: セグメント境界の正確な指定が不可能
3. **`/v1/completions` で `input_ids` を渡してもキャッシュのロードが発生しない**: ストアは行われるがリユースなし
4. **vLLM GPUワーカーへのパッチが必要**: `gpu_worker.py` の手動修正が必要で、バージョン間で互換性が壊れる

**ワークアラウンド**（2026年2月3日 @rick-heig）: Qwen3ファミリーでは `<|file_sep|>` のような常に同じトークンIDに変換される特殊トークンをblend区切り文字として使うことで、テキストレベルでの操作が可能になる。

### 3. CacheBlend V1の品質・安定性バグ

- **ガーブル出力** ([#2496](https://github.com/LMCache/LMCache/issues/2496), open): 複数エントリ処理時、最初以外の出力が文字化け。GPU/CPUメモリのクリア不足の疑い
- **キャッシュヒット後の保存漏れ** ([#2029](https://github.com/LMCache/LMCache/issues/2029), open/stale): 部分ヒット時、新規計算トークンが保存されない → 後続リクエストのヒット率低下
- **先頭ミス時の探索打ち切り** ([#2029](https://github.com/LMCache/LMCache/issues/2029)): 最初のチャンクがキャッシュミスすると後続チャンクの探索なし → RAGで致命的
- **layerwiseモードでのKVキャッシュ破損** (PR#2329コメント, 2026-02-03): layerwise有効時、温度0でも異なる応答

### 4. LMCache側の安定化PR

- **[PR#762](https://github.com/LMCache/LMCache/pull/762)** (2025-06マージ): CacheBlend V1初期実装
- **[PR#2329](https://github.com/LMCache/LMCache/pull/2329)** (open, コンフリクトあり): layerwise/blendingエッジケース修正 + vLLMパッチヘルパー。テスト中にlayerwiseのKVキャッシュ破損が新たに報告されており、マージ見通し不明

### 5. バージョン互換性

LMCache-Ascendチームが整理した互換性マトリクス ([#154](https://github.com/LMCache/LMCache-Ascend/issues/154)):

| vLLM | LMCache | CacheBlend |
|------|---------|-----------|
| 0.9.2 | 0.3.3 / 0.3.7 | Production Ready |
| 0.10.0 | 0.3.7 | Not supported |
| 0.11.0 | 0.3.7 | Not supported |
| 0.10.0 | 0.3.12 | Production Ready |
| 0.11.0 | 0.3.12 | Production Ready |

※ Ascend NPU版マトリクス。GPU版LMCacheとは異なる可能性あり。

## タイムライン

| 日付 | イベント |
|------|---------|
| 2025-06-07 | CacheBlend V1 初期実装が LMCache にマージ ([PR#762](https://github.com/LMCache/LMCache/pull/762)) |
| 2025-07-24 | オンライン推論での動作不良が初報告 ([#1136](https://github.com/LMCache/LMCache/issues/1136)) |
| 2025-08-05 | トークン化不一致問題の根本原因が特定 |
| 2025-09-30 | vLLM本体に Generalized KV Cache Reuse RFC提出 ([#25950](https://github.com/vllm-project/vllm/issues/25950)) |
| 2025-10-31 | `vllm serve` サポートの明確な Feature Request ([#1936](https://github.com/LMCache/LMCache/issues/1936)) |
| 2025-12-29 | layerwise/blending修正PR提出 ([PR#2329](https://github.com/LMCache/LMCache/pull/2329)) |
| 2026-01-12 | vLLM RFC#25950にてサブリクエスト分割アプローチ発見の報告 |
| 2026-01-27 | ガーブル出力バグ報告 ([#2496](https://github.com/LMCache/LMCache/issues/2496)) |
| 2026-02-03 | layerwise KVキャッシュ破損の新規報告 + 特殊トークンワークアラウンド提案 |

## 結論・所見

### オンライン推論（`vllm serve`）の現状

**動作しない**。オフライン専用の状態が約8ヶ月続いている。根本的な問題はCacheBlendのセグメント区切りがトークンレベルの精密な制御を要求するのに対し、HTTP APIがテキストレベルの入力しか受け付けない点にある。

2つのアプローチが存在するが、いずれも未完成:

1. **LMCache側のアプローチ**: vLLMのworkerにパッチを当てて対応。バージョン間の互換性維持が困難
2. **vLLM本体側のアプローチ**: RFC#25950のサブリクエスト分割方式。コード未公開。実現すればvLLM本体の機能としてCacheBlendが使えるようになる可能性があるが、不透明

### プラグイン開発への示唆

CacheBlendの現状は、vLLMのKV Connectorインターフェースの限界を示している:
- 現在のKV ConnectorはPrefix Caching前提（連続ブロックの転送）
- 非連続キャッシュ再利用にはScheduler・Attention Kernelレベルの変更が必要
- RFC#25950のサブリクエスト分割アプローチが実現すれば、KV Connector層のみで対応可能になる

独自プラグイン作成を検討する場合、CacheBlendの統合方式の行方は重要な参考情報となる。
