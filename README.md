# coderead: OSSコードリーディングフレームワーク

Claude Codeを活用してOSSのコードリーディングを構造化し、知見をmdbookで公開するためのフレームワーク。

## セットアップ

### 前提条件

- [uv](https://docs.astral.sh/uv/)
- [mdbook](https://rust-lang.github.io/mdBook/guide/installation.html)
- [mdbook-mermaid](https://github.com/badboy/mdbook-mermaid)

```bash
# mdbook と mdbook-mermaid のインストール
cargo install mdbook mdbook-mermaid

# または Homebrew（macOS）
brew install mdbook
cargo install mdbook-mermaid
```

### テンプレートからプロジェクト作成

```bash
# テンプレートリポジトリからクローン
gh repo create coderead-{対象OSS} --template {テンプレートリポジトリURL}
cd coderead-{対象OSS}

# 対象OSSをsubmoduleとして追加
git submodule add {対象OSSのURL} target/{OSS名}

# 依存関係のインストール
uv sync

# 初期セットアップ
uv run task setup
```

### 初期設定

1. `.claude/CLAUDE.md` の「プロジェクト概要」セクションを記入
2. `docs/book.toml` の `title` を更新
3. `docs/src/README.md` を対象OSSに合わせて編集

## 使い方

### コードリーディング（Claude Code）

```bash
# Claude Code を起動してコードリーディングを開始
claude
```

Claude Codeは `.claude/CLAUDE.md` のプロトコルに従って調査を進めます。

### ドキュメント

```bash
# ローカルプレビュー
uv run task docs-serve

# ビルド
uv run task docs-build

# GitHub Pages へデプロイ
uv run task docs-deploy
```

### ユーティリティ

```bash
# SUMMARY.md を自動生成
uv run task summary

# リンク検証
uv run task validate

# 調査進捗の統計表示
uv run task stats
```

## ディレクトリ構成

```
.claude/CLAUDE.md          # Claude Code用プロジェクト指示書
target/                    # 対象OSSソースコード（git submodule）
docs/
├── book.toml              # mdbook設定
└── src/                   # ドキュメントソース
    ├── SUMMARY.md         # 目次（自動生成可）
    ├── architecture/      # アーキテクチャ
    ├── components/        # コンポーネント別分析
    ├── glossary.md        # 用語集
    └── appendix/          # 付録
.state/                    # 調査状態管理
├── exploration-log.md     # 探索進捗（集約）
├── context-index.md       # ドキュメント索引
├── next-actions.md        # 次回アクション
├── reading-guide.md       # 対象OSS固有の読解ルール・ユーザー優先度
└── sessions/              # セッション記録
templates/                 # ドキュメントテンプレート
scripts/                   # ユーティリティスクリプト
```

## 調査フェーズ

| フェーズ                      | 目的             | 成果物                                    |
| ----------------------------- | ---------------- | ----------------------------------------- |
| Phase 0: オリエンテーション   | 全体像の把握     | `architecture/overview.md`, `glossary.md`, `.state/reading-guide.md` |
| Phase 1: 垂直スライス         | 主要フローの追跡 | `architecture/data-flow.md`               |
| Phase 2: コンポーネント深堀り | 個別分析         | `components/*/summary.md`                 |
| Phase 3: 横断的機能           | 機能横断の理解   | 各ドキュメントの充実                      |

## ドキュメント規約

### 深度マーカー

- `[SHALLOW]` - 概要レベル
- `[MEDIUM]` - API・データフローを理解
- `[DEEP]` - 内部実装まで理解

### 確信度マーカー

- `[VERIFIED]` - ソースコードで確認済み
- `[INFERRED]` - 推測を含む
- `[TODO]` - 未調査
