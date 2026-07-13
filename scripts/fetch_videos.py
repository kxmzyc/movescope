"""通过 yt-dlp 搜索或下载已获授权的训练视频。"""

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
        epilog="仅可下载和处理你拥有或已获得授权的视频。",
    )
    parser.add_argument("--action", default="squat", help="动作标识，例如 squat")
    parser.add_argument("--mode", choices=["expert", "test"], required=True)
    parser.add_argument("--n", type=int, default=3, help="每个搜索词返回的结果数量")
    parser.add_argument("--lang", choices=["zh", "en", "both"], default="zh")
    parser.add_argument("--output-dir", help="指定下载目录，默认使用 data/{mode}/{action}。")
    parser.add_argument("--dry-run", action="store_true", help="只显示候选标题和 URL，不下载视频")
    return parser.parse_args()


def selected_terms(action: str, mode: str, lang: str) -> list[str]:
    languages = ["zh", "en"] if lang == "both" else [lang]
    terms: list[str] = []
    for language in languages:
        terms.extend(SEARCH_TERMS.get((action, mode, language), []))
    if not terms:
        raise SystemExit(f"未配置搜索词：action={action!r}，mode={mode!r}，lang={lang!r}")
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
        print(f"[失败] {term}：{result.stderr.strip()}")
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
    print(f"[下载] {term}：成功 {downloaded} 个，失败 {failed} 个")
    return downloaded, failed


def main() -> None:
    args = parse_args()
    if not yt_dlp_available():
        raise SystemExit("当前 Python 环境未安装 yt-dlp，请运行：pip install yt-dlp")

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

    print(f"处理完成：成功 {success} 个，失败 {failed} 个")


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
