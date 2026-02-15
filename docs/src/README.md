# コードリーディング: vLLM & LMCache

LLM推論サービングに関連するOSSのコードリーディング結果を構造化して蓄積するプロジェクト。

## 対象OSS

| OSS | ソースコード | 概要 | Phase |
|-----|-------------|------|-------|
| [vLLM](vllm/README.md) | `target/vllm/` | LLM推論サービングエンジン | Phase 2 |
| [LMCache](lmcache/README.md) | `target/LMCache/` | KVキャッシュ保存・共有・再利用ライブラリ | Phase 0a |

## プロジェクト横断

- [横断調査](cross-project/README.md) — 複数OSSにまたがる調査報告
