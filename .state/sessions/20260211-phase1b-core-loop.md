# セッション記録: コアループ追跡

> **日付**: 2026-02-11
> **フェーズ**: Phase 1（垂直スライス セッション2/3）

## 目的

EngineCore.step() を起点にコアループ（Scheduler, KVCacheManager）の動作を追跡し、schedule → execute → update サイクルの全体像を把握する。

## Phase完了条件の進捗

- [x] data-flow.md にエンドツーエンドのフローが記載されている（上流パス + コアループ。下流パスは次セッション）
- [x] フロー上の全コンポーネントが `docs/src/components/` に登録されている（engine-core, scheduler, kv-cache-manager を追加）
- [ ] 各コンポーネントの優先度が決定されている → セッション3で確定

## 調査経路

1. EngineCore.step() (core.py L389-422) — schedule→execute→update サイクルの全体像
2. EngineCore.__init__() (core.py L82-224) — Scheduler, ModelExecutor, KVキャッシュ初期化
3. Scheduler.schedule() (scheduler.py L321-896) — 3フェーズ構成の発見
4. KVCacheManager.allocate_slots() (kv_cache_manager.py L206-376) — ブロック配置図の理解
5. KVCacheManager.get_computed_blocks() (kv_cache_manager.py L164-204) — プレフィックスキャッシュメカニズム
6. BlockPool (block_pool.py L128) — 物理ブロック管理、参照カウント、LRU Eviction
7. SchedulerOutput (output.py L184-253) — 全15フィールドの整理
8. ModelRunnerOutput (outputs.py L160-196) — 全10フィールドの整理
9. update_from_output() (scheduler.py L1241-1494) — 出力処理と完了判定
10. Request (request.py) — ステータス遷移図の作成

## 発見

### EngineCore
- step() は `tuple[dict[int, EngineCoreOutputs], bool]` を返す（クライアントインデックス別出力 + 実行フラグ）
- batch_queue による パイプライン並列化（max_concurrent_batches > 1 時）
- model_output=None 時に sample_tokens() が呼ばれる（async_scheduling 分離）
- KVキャッシュ初期化: get_kv_cache_specs → determine_available_memory → generate_scheduler_kv_cache_config

### Scheduler
- **Unified Compute Model**: Prefill/Decode を区別しない統一的なスケジューリング
- schedule() は3フェーズ: RUNNING処理(L350-517) → WAITING処理(L532-800) → Output構築(L827-896)
- トークン予算: `max_num_scheduled_tokens` で初期化、単調減少
- プリエンプション: RUNNINGのみ発生、Priority/FIFO ポリシー、WAITINGキュー先頭に戻す
- Request ステータス: WAITING → (WAITING_FOR_FSM/REMOTE_KVS/STREAMING_REQ) → RUNNING → FINISHED_*/PREEMPTED
- NewRequestData（フルデータ）vs CachedRequestData（差分のみ）でプロセス間通信を最適化
- KV Transfer統合: load_kv_async, delay_cache_blocks フラグ

### KVCacheManager
- 4層階層: KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool
- ブロック配置: comp | new_comp | ext_comp | new | lookahead
- allocate_slots() 失敗時 None を返す → プリエンプション誘発（RUNNING）またはスキップ（WAITING）
- プレフィックスキャッシュ: BlockHashToBlockMap でハッシュチェーン検索、最長一致
- 参照カウント方式: ref_cnt=0 で空きキュー、touch()で増加、逆順解放でLRU効率化
- スライディングウィンドウ: null_block で置換して物理メモリ解放

## つまずき・未解決

- batch_queue パイプライン並列化（step_with_batch_queue）の詳細動作は深追いしていない
- プリエンプション発生のメモリ圧力閾値が具体的にどう決まるかは不明
- async_scheduling と Speculative Decoding のドラフトトークンタイミング相互作用の詳細

## 成果物

- `docs/src/components/engine-core/summary.md` [MEDIUM] — 新規作成
- `docs/src/components/scheduler/summary.md` [MEDIUM] — 新規作成
- `docs/src/components/kv-cache-manager/summary.md` [MEDIUM] — 新規作成
- `docs/src/architecture/data-flow.md` — コアループセクション追加、SchedulerOutput/ModelRunnerOutput [TODO]解消、深度 [SHALLOW]→[MEDIUM]
- `docs/src/glossary.md` — Unified Compute Model 追加

## Phase完了判定

- このPhaseの完了条件を全て満たしたか: No
- 残り: セッション3（下流パス追跡、コンポーネント優先度確定）

## 次回への引き継ぎ

Phase 1 セッション3: 下流パス追跡（Executor → Worker → GPUModelRunner → OutputProcessor）
- Executor/Worker の委譲パターン把握
- GPUModelRunner.execute_model() の概要
- OutputProcessor → RequestOutput 変換とDetokenize
- data-flow.md 完成、コンポーネント優先度確定
