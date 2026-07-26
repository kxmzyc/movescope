"""MoveScope 命令行工具集。

每个模块提供一个 main() 入口，通过 pyproject.toml 的
[project.scripts] 注册为 movescope-* 命令；scripts/ 目录下保留
同名薄转发壳以兼容 `python scripts/xxx.py` 的历史调用方式。
"""
