#!/usr/bin/env python3
"""
docs/src/SUMMARY.md を docs/src/ のディレクトリ構造から自動生成する。

マルチOSS対応:
- docs/src/vllm/ と docs/src/lmcache/ を各OSSセクションとして走査
- docs/src/cross-project/ をプロジェクト横断セクションとして走査
- 各OSS内のサブセクション（architecture/components/investigations）を維持
"""

from pathlib import Path


DOCS_SRC = Path(__file__).parent.parent / "docs" / "src"

HEADER = """# Summary

[はじめに](README.md)
"""

FOOTER = ""

# 走査から除外するファイル/ディレクトリ
EXCLUDE_FILES = {"README.md", "SUMMARY.md", "glossary.md"}
EXCLUDE_DIRS: set[str] = set()

# OSSプロジェクト定義
OSS_PROJECTS = [
    {
        "dir": "vllm",
        "title": "vLLM",
        "subsections": [
            ("architecture", "アーキテクチャ"),
            ("components", "コンポーネント"),
            ("investigations", "調査報告"),
        ],
        "appendix": [
            ("glossary.md", "用語集"),
            ("appendix/file-index.md", "ファイル索引"),
        ],
    },
    {
        "dir": "lmcache",
        "title": "LMCache",
        "subsections": [
            ("architecture", "アーキテクチャ"),
            ("components", "コンポーネント"),
            ("investigations", "調査報告"),
        ],
        "appendix": [
            ("glossary.md", "用語集"),
        ],
    },
]


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
    name = filepath.stem
    return name.replace("-", " ").replace("_", " ").title()


def scan_directory(
    dirpath: Path, base: Path, depth: int = 0
) -> list[str]:
    """ディレクトリを再帰的に走査してSUMMARY.mdのエントリを生成する。"""
    lines: list[str] = []
    indent = "  " * depth

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

        sub_lines = scan_directory(d, base, depth + 1)
        lines.extend(sub_lines)

    return lines


def generate_oss_section(project: dict) -> list[str]:
    """1つのOSSプロジェクトのセクションを生成する。"""
    parts: list[str] = []
    oss_dir = DOCS_SRC / project["dir"]
    if not oss_dir.exists():
        return parts

    # OSSトップレベルのリンク
    readme = oss_dir / "README.md"
    if readme.exists():
        title = get_title(readme)
        parts.append(f"\n---\n\n# {project['title']}\n")
        parts.append(f"- [{title}]({project['dir']}/README.md)")
    else:
        parts.append(f"\n---\n\n# {project['title']}\n")

    # サブセクション
    for subdir_name, section_title in project["subsections"]:
        subdir = oss_dir / subdir_name
        if not subdir.exists():
            continue
        lines = scan_directory(subdir, DOCS_SRC)
        if lines:
            parts.append(f"\n# {project['title']}: {section_title}\n")
            parts.append("\n".join(lines))

    # 付録
    appendix_items = []
    for rel_path, label in project.get("appendix", []):
        full_path = oss_dir / rel_path
        if full_path.exists():
            appendix_items.append(f"- [{label}]({project['dir']}/{rel_path})")
    if appendix_items:
        parts.append(f"\n# {project['title']}: 付録\n")
        parts.append("\n".join(appendix_items))

    return parts


def generate_summary() -> str:
    """SUMMARY.md の内容を生成する。"""
    parts = [HEADER]

    # 各OSSプロジェクトのセクション
    for project in OSS_PROJECTS:
        oss_parts = generate_oss_section(project)
        parts.extend(oss_parts)

    # プロジェクト横断セクション
    cross_dir = DOCS_SRC / "cross-project"
    if cross_dir.exists():
        lines = scan_directory(cross_dir, DOCS_SRC)
        if lines:
            parts.append("\n\n---\n\n# プロジェクト横断\n")
            parts.append("\n".join(lines))

    if FOOTER:
        parts.append(FOOTER)

    return "\n".join(parts) + "\n"


def main():
    summary_path = DOCS_SRC / "SUMMARY.md"
    content = generate_summary()

    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
        if existing == content:
            print("SUMMARY.md: 変更なし")
            return

    summary_path.write_text(content, encoding="utf-8")
    print(f"SUMMARY.md: 更新しました ({summary_path})")


if __name__ == "__main__":
    main()
