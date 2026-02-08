#!/usr/bin/env python3
"""
docs/src/ 内のMarkdownファイルのリンクを検証する。

チェック項目:
- 内部リンク（相対パス）のファイルが存在するか
- target/ へのファイル参照が存在するか
"""

from pathlib import Path
import re
import sys


DOCS_SRC = Path(__file__).parent.parent / "docs" / "src"
PROJECT_ROOT = Path(__file__).parent.parent

# Markdownリンクのパターン: [text](path) or [text](path#anchor)
LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

# ファイル参照パターン: `target/path/to/file.py:123`
FILE_REF_PATTERN = re.compile(r"`(target/[^`]+?)(?::(\d+))?`")


def check_links(filepath: Path) -> list[str]:
    """Markdownファイル内のリンクを検証する。"""
    errors = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [f"{filepath}: 読み取りエラー: {e}"]

    for i, line in enumerate(content.splitlines(), 1):
        # Markdownリンク
        for match in LINK_PATTERN.finditer(line):
            link_text, link_path = match.groups()
            # 外部リンク、空リンク、アンカーのみはスキップ
            if (
                link_path.startswith("http")
                or link_path.startswith("#")
                or link_path == ""
                or link_path == "()"
            ):
                continue
            # アンカー部分を除去
            clean_path = link_path.split("#")[0]
            if not clean_path:
                continue
            # 相対パスを解決
            target = (filepath.parent / clean_path).resolve()
            if not target.exists():
                errors.append(
                    f"{filepath}:{i}: リンク切れ: [{link_text}]({link_path})"
                )

        # ファイル参照（`target/...`）
        for match in FILE_REF_PATTERN.finditer(line):
            ref_path = match.group(1)
            target = (PROJECT_ROOT / ref_path).resolve()
            if not target.exists():
                errors.append(
                    f"{filepath}:{i}: ファイル参照が見つからない: `{ref_path}`"
                )

    return errors


def main():
    all_errors = []

    # docs/src/ 内の全Markdownファイルを検証
    for md_file in sorted(DOCS_SRC.rglob("*.md")):
        errors = check_links(md_file)
        all_errors.extend(errors)

    # .state/ 内も検証
    state_dir = PROJECT_ROOT / ".state"
    if state_dir.exists():
        for md_file in sorted(state_dir.rglob("*.md")):
            errors = check_links(md_file)
            all_errors.extend(errors)

    if all_errors:
        print(f"検証エラー: {len(all_errors)} 件\n")
        for error in all_errors:
            print(f"  ✗ {error}")
        sys.exit(1)
    else:
        print("✓ すべてのリンクが有効です")


if __name__ == "__main__":
    main()
