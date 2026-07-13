"""检查 MoveScope 本地开发环境。"""

from __future__ import annotations

import importlib.util
from importlib import metadata
import shutil
import subprocess
import sys


COMMANDS = ["python", "node", "yt-dlp"]
PYTHON_PACKAGES = ["cv2", "mediapipe", "numpy"]
PACKAGE_DISTRIBUTIONS = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def command_version(command: str) -> str:
    if command == "yt-dlp":
        return module_version("yt_dlp")

    if shutil.which(command) is None:
        return "未安装"

    args = [command, "--version"]
    if command == "python":
        args = [command, "-c", "import sys; print(sys.version.split()[0])"]

    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return f"错误：{result.stderr.strip()}"
    return result.stdout.strip().splitlines()[0]


def module_version(module_name: str) -> str:
    if importlib.util.find_spec(module_name) is None:
        return "未安装"
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"错误：{result.stderr.strip()}"
    return result.stdout.strip().splitlines()[0]


def package_status(package: str) -> str:
    if importlib.util.find_spec(package) is None:
        return "未安装"
    try:
        return metadata.version(PACKAGE_DISTRIBUTIONS.get(package, package))
    except metadata.PackageNotFoundError:
        return "已安装"


def main() -> None:
    print("命令行工具：")
    for command in COMMANDS:
        print(f"  {command}: {command_version(command)}")

    print("\nPython 包：")
    for package in PYTHON_PACKAGES:
        print(f"  {package}: {package_status(package)}")

    if sys.version_info >= (3, 13):
        print("\n警告：当前使用 Python 3.13。MediaPipe 0.10.x 请使用 Python 3.10 或 3.11。")


if __name__ == "__main__":
    main()
