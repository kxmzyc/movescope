"""Check local prerequisites for MoveScope development."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


COMMANDS = ["python", "node", "yt-dlp"]
PYTHON_PACKAGES = ["cv2", "mediapipe", "numpy"]


def command_version(command: str) -> str:
    if command == "yt-dlp":
        return module_version("yt_dlp")

    if shutil.which(command) is None:
        return "missing"

    args = [command, "--version"]
    if command == "python":
        args = [command, "-c", "import sys; print(sys.version.split()[0])"]

    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return f"error: {result.stderr.strip()}"
    return result.stdout.strip().splitlines()[0]


def module_version(module_name: str) -> str:
    if importlib.util.find_spec(module_name) is None:
        return "missing"
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"error: {result.stderr.strip()}"
    return result.stdout.strip().splitlines()[0]


def package_status(package: str) -> str:
    if importlib.util.find_spec(package) is None:
        return "missing"
    module = __import__(package)
    return getattr(module, "__version__", "installed")


def main() -> None:
    print("Commands:")
    for command in COMMANDS:
        print(f"  {command}: {command_version(command)}")

    print("\nPython packages:")
    for package in PYTHON_PACKAGES:
        print(f"  {package}: {package_status(package)}")

    if sys.version_info >= (3, 13):
        print("\nWARNING: Python 3.13 is active. Use Python 3.10 or 3.11 for MediaPipe 0.10.x.")


if __name__ == "__main__":
    main()
