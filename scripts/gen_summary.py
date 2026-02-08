#!/usr/bin/env python3
"""
docs/src/SUMMARY.md を docs/src/ のディレクトリ構造から自動生成する。

ルール:
- docs/src/ 配下の .md ファイルを走査
- ディレクトリ内に summary.md があればセクションのトップとして扱う
- README.md, SUMMARY.md は特別扱い
- ファイル名からタイトルを推測（H1見出しがあればそれを使用）
"""

from pathlib import Path


DOCS_SRC = Path(__file__).parent.parent / "docs" / "src"

# 固定セクション（SUMMARY.mdの先頭部分）
HEADER = """# Summary

[はじめに](README.md)
"""

# 固定セクション（SUMMARY.mdの末尾部分）
FOOTER = """
---

# 付録

- [用語集](glossary.md)
- [ファイル索引](appendix/file-index.md)
"""

# 走査から除外するファイル/ディレクトリ
EXCLUDE_FILES = {"README.md", "SUMMARY.md", "glossary.md"}
EXCLUDE_DIRS = {"appendix", "investigations"}


def get_title(filepath: Path) -> str:
    """Markdownファイルの最初のH1見出しをタイトルとして取得する。"""
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
    except (OSError, UnicodeDecodeError):
        pass
    # H1が見つからなければファイル名を整形して返す
    name = filepath.stem
    return name.replace("-", " ").replace("_", " ").title()


def scan_directory(
    dirpath: Path, base: Path, depth: int = 0
) -> list[str]:
    """ディレクトリを再帰的に走査してSUMMARY.mdのエントリを生成する。"""
    lines: list[str] = []
    indent = "  " * depth

    # summary.md があればディレクトリのトップエントリとして扱う
    summary_file = dirpath / "summary.md"

    # ディレクトリ内のファイルとサブディレクトリをソート
    entries = sorted(dirpath.iterdir())
    md_files = [
        e for e in entries
        if e.is_file()
        and e.suffix == ".md"
        and e.name not in EXCLUDE_FILES
        and e.name != "summary.md"
    ]
    subdirs = [
        e for e in entries
        if e.is_dir() and e.name not in EXCLUDE_DIRS
    ]

    for f in md_files:
        rel = f.relative_to(base)
        title = get_title(f)
        lines.append(f"{indent}- [{title}]({rel})")

    for d in subdirs:
        dir_summary = d / "summary.md"
        dir_rel = dir_summary.relative_to(base) if dir_summary.exists() else None
        dir_title = get_title(dir_summary) if dir_summary.exists() else d.name.replace("-", " ").title()

        if dir_rel:
            lines.append(f"{indent}- [{dir_title}]({dir_rel})")
        else:
            lines.append(f"{indent}- [{dir_title}]()")

        # サブディレクトリ内のファイルを再帰的に走査
        sub_lines = scan_directory(d, base, depth + 1)
        lines.extend(sub_lines)

    return lines


def generate_summary() -> str:
    """SUMMARY.md の内容を生成する。"""
    parts = [HEADER]

    # アーキテクチャセクション
    arch_dir = DOCS_SRC / "architecture"
    if arch_dir.exists():
        parts.append("\n---\n\n# アーキテクチャ\n")
        arch_lines = scan_directory(arch_dir, DOCS_SRC)
        parts.append("\n".join(arch_lines))

    # コンポーネントセクション
    comp_dir = DOCS_SRC / "components"
    if comp_dir.exists():
        parts.append("\n\n---\n\n# コンポーネント\n")
        comp_lines = scan_directory(comp_dir, DOCS_SRC)
        parts.append("\n".join(comp_lines))

    # 調査報告セクション
    inv_dir = DOCS_SRC / "investigations"
    if inv_dir.exists():
        inv_lines = scan_directory(inv_dir, DOCS_SRC)
        if inv_lines:
            parts.append("\n\n---\n\n# 調査報告\n")
            parts.append("\n".join(inv_lines))

    parts.append(FOOTER)

    return "\n".join(parts) + "\n"


def main():
    summary_path = DOCS_SRC / "SUMMARY.md"
    content = generate_summary()

    # 既存の内容と比較して変更がある場合のみ書き込み
    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
        if existing == content:
            print("SUMMARY.md: 変更なし")
            return

    summary_path.write_text(content, encoding="utf-8")
    print(f"SUMMARY.md: 更新しました ({summary_path})")


if __name__ == "__main__":
    main()
