"""Search or download authorized exercise videos via yt-dlp."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SEARCH_TERMS = {
    ("squat", "expert", "zh"): ["标准深蹲教学", "深蹲正确姿势示范", "深蹲标准动作"],
    ("squat", "expert", "en"): [
        "perfect squat form tutorial",
        "squat technique guide",
        "proper squat depth",
    ],
    ("squat", "test", "zh"): ["健身深蹲", "深蹲训练", "深蹲错误示例"],
    ("squat", "test", "en"): ["squat workout", "squat form check", "squat mistakes"],
}


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use only videos you own or are authorized to download and process.",
    )
    parser.add_argument("--action", default="squat", help="Action name, for example: squat")
    parser.add_argument("--mode", choices=["expert", "test"], required=True)
    parser.add_argument("--n", type=int, default=3, help="Number of results per search term")
    parser.add_argument("--lang", choices=["zh", "en", "both"], default="zh")
    parser.add_argument("--output-dir", help="Override download directory. Defaults to data/{mode}/{action}.")
    parser.add_argument("--dry-run", action="store_true", help="Only print candidate titles and URLs")
    return parser.parse_args()


def selected_terms(action: str, mode: str, lang: str) -> list[str]:
    languages = ["zh", "en"] if lang == "both" else [lang]
    terms: list[str] = []
    for language in languages:
        terms.extend(SEARCH_TERMS.get((action, mode, language), []))
    if not terms:
        raise SystemExit(f"No search terms configured for action={action!r}, mode={mode!r}, lang={lang!r}")
    return terms


def run_yt_dlp(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "yt_dlp", *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def dry_run(term: str, n: int) -> tuple[int, int]:
    query = f"ytsearch{n}:{term}"
    result = run_yt_dlp([query, "--print", "%(title)s\t%(webpage_url)s\t%(duration_string)s"])
    if result.returncode != 0:
        print(f"[failed] {term}: {result.stderr.strip()}")
        return 0, 1

    count = 0
    for line in result.stdout.splitlines():
        if line.strip():
            print(line)
            count += 1
    return count, 0


def download(term: str, action: str, mode: str, n: int, output_dir: Path | None = None) -> tuple[int, int]:
    output_dir = output_dir or Path("data") / mode / action
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for ext in VIDEO_EXTENSIONS for path in output_dir.glob(f"*{ext}")}
    output_template = str(output_dir / "%(id)s.%(ext)s")
    result = run_yt_dlp(
        [
            f"ytsearch{n}:{term}",
            "--max-downloads",
            str(n),
            "--match-filter",
            "duration>15 & duration<120",
            "--format",
            "best[height>=480]/best",
            "-o",
            output_template,
        ]
    )

    if result.returncode != 0:
        Path("data").mkdir(exist_ok=True)
        with Path("data/download_errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {term} ===\n{result.stderr}\n")

    after = {path.name for ext in VIDEO_EXTENSIONS for path in output_dir.glob(f"*{ext}")}
    downloaded = len(after - before)
    failed = 0 if result.returncode == 0 else 1
    print(f"[download] {term}: downloaded={downloaded}, failed={failed}")
    return downloaded, failed


def main() -> None:
    args = parse_args()
    if not yt_dlp_available():
        raise SystemExit("yt-dlp is not installed in this Python environment. Install it with: pip install yt-dlp")

    success = 0
    failed = 0
    for term in selected_terms(args.action, args.mode, args.lang):
        if args.dry_run:
            ok, bad = dry_run(term, args.n)
        else:
            ok, bad = download(
                term,
                args.action,
                args.mode,
                args.n,
                Path(args.output_dir) if args.output_dir else None,
            )
        success += ok
        failed += bad

    print(f"Done. success={success}, failed={failed}")


def yt_dlp_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


if __name__ == "__main__":
    main()
