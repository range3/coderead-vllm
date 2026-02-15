# Phase 1a: 上流パス追跡

**日付**: 2026-02-09
**Phase**: 1 (垂直スライス セッション1/3)
**テーマ**: テキスト推論フルパスの上流パス（API → AsyncLLM → InputProcessor → EngineCore到達）

## 調査内容

テキスト推論リクエストがユーザー入力からEngineCoreに到達するまでのフローを追跡した。

### 読んだソースコード

| ファイル | 読んだ範囲 | 発見 |
|---------|---------|------|
| `async_llm.py` | L71-190, L286-430, L537-716 | generate()はAsyncGenerator。add_request()でInputProcessor→OutputProcessor→EngineCoreの3段階 |
| `input_processor.py` | L56-100, L521-671 | process_inputs()でトークナイズ→パラメータ正規化→EngineCoreRequest構築。テキスト推論ではmm_features=None |
| `__init__.py` | L1-200 | EngineCoreRequest(msgspec.Struct, array_like)。EngineCoreOutput, EngineCoreOutputsの定義 |
| `core_client.py` | L63-200, L442-521, L822-951 | MPClient: ZMQ ROUTER(input) + PULL(output)。msgpackシリアライゼーション。weakrefで循環参照回避 |
| `core.py` | L79-180, L288-327, L389-422 | EngineCore.add_request()→scheduler.add_request()。step()でschedule→execute→update |
| `llm.py` | L396-458, L1850-1949 | LLM.generate()は_run_completion→_add_request×N→_run_engine()。同期ポーリング |

### 主要な発見

1. **リクエストID管理**: 外部IDに8文字ランダムサフィックスを付与して内部IDとする。外部IDはexternal_req_idフィールドに退避
2. **ZMQソケット型**: input_socket=ROUTER, output_socket=PULL（Explore agent報告のPUSH/PULLとは異なる）
3. **output_handler**: asyncioバックグラウンドタスクとしてEngineCoreの出力を継続的に受信し、RequestOutputCollectorキューにpush
4. **SamplingParams正規化**: InputProcessorがclone()してmax_tokens設定、generation_config反映、tokenizer反映を実施
5. **n>1サンプリング**: ParentRequestクラスでファンアウト管理。子リクエストはIDサフィックスで区別
6. **EngineCoreRequest**: msgspec.Struct(array_like, omit_defaults)で効率的にmsgpackシリアライズ

### 新たな疑問

- InputPreprocessor内部のトークナイザ呼び出しフロー詳細
- n>1サンプリング時のParentRequest/子リクエスト管理の仕組み

### 解決した疑問

- ZMQ IPCを採用した理由: GIL回避とスケジューリング/GPU実行の並行実現

## 成果物

- `docs/src/architecture/data-flow.md` — 骨格作成（全体Mermaid図 + 上流パス詳細 + 境界データ構造）
- `docs/src/components/entrypoint/summary.md` — [SHALLOW]
- `docs/src/components/input-processor/summary.md` — [SHALLOW]
- `docs/src/components/engine-core-client/summary.md` — [SHALLOW]

## 次回

Phase 1 セッション2: コアループ追跡（EngineCore → Scheduler → KVCacheManager → Executor）
