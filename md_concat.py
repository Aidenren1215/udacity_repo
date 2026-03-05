from __future__ import annotations

from pathlib import Path
import re
from datetime import datetime
from typing import List, Tuple, Optional

DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

def extract_date_from_name(filename: str) -> Optional[datetime]:
    """
    Extract YYYY-MM-DD from filename. Return None if not found.
    """
    m = DATE_RE.search(filename)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return datetime(y, mo, d)

def list_md_files_by_year(root: Path) -> List[Path]:
    """
    root/
      2014/
      2015/
      ...
    Return all .md files under year folders.
    """
    files: List[Path] = []
    for year_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]):
        files.extend(sorted(year_dir.glob("*.md")))
    return files

def filter_and_sort(files: List[Path], *, subcommittee: bool) -> List[Path]:
    """
    Split into ALCO minutes vs Sub-committee minutes by filename keyword,
    then sort by date extracted from filename; fall back to name sort.
    """
    if subcommittee:
        kept = [f for f in files if "Sub-committee" in f.name]
    else:
        kept = [f for f in files if "Sub-committee" not in f.name and "ALCO minutes" in f.name]

    # Sort by extracted date, then by filename for stability
    def key(p: Path):
        dt = extract_date_from_name(p.name)
        # Put undated files at the end
        return (dt is None, dt or datetime.max, p.name)

    return sorted(kept, key=key)

def concat_markdown(files: List[Path]) -> str:
    """
    Concatenate files with clear separators and source markers.
    """
    parts: List[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        dt = extract_date_from_name(f.name)
        date_str = dt.strftime("%Y-%m-%d") if dt else "UNKNOWN_DATE"

        parts.append(
            "\n".join([
                "",
                "---",
                f"# SOURCE: {f.as_posix()}",
                f"# DATE: {date_str}",
                "---",
                "",
                text,
                ""
            ])
        )
    return "\n".join(parts).lstrip()

def build_concat_for_root(root_dir: str | Path) -> Tuple[str, str]:
    """
    Return (alco_minutes_concat, subcommittee_concat).
    """
    root = Path(root_dir)
    all_files = list_md_files_by_year(root)

    alco_files = filter_and_sort(all_files, subcommittee=False)
    sub_files  = filter_and_sort(all_files, subcommittee=True)

    alco_concat = concat_markdown(alco_files)
    sub_concat  = concat_markdown(sub_files)

    return alco_concat, sub_concat


if __name__ == "__main__":
    root = Path("files/alco minutes md")  # 改成你的实际路径
    alco_concat, sub_concat = build_concat_for_root(root)

    # 你想落盘就写文件
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "SG_ALCO_minutes_ALL_YEARS.concat.md").write_text(alco_concat, encoding="utf-8")
    (out_dir / "SG_ALCO_subcommittee_minutes_ALL_YEARS.concat.md").write_text(sub_concat, encoding="utf-8")

    print("ALCO minutes files concat length:", len(alco_concat))
    print("Sub-committee files concat length:", len(sub_concat))
    print("Wrote to:", out_dir.resolve())