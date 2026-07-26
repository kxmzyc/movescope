"""MoveScope 领域异常体系。

领域异常同时继承 MoveScopeError 和最接近其语义的内建异常
（FileNotFoundError / ValueError），因此既可以按领域类型精确捕获，
也不破坏仍按内建异常捕获的既有调用方。
"""

from __future__ import annotations


class MoveScopeError(Exception):
    """所有 MoveScope 领域异常的公共基类。"""


class TemplateNotFoundError(MoveScopeError, FileNotFoundError):
    """请求的动作模板文件不存在。"""


class InvalidInputError(MoveScopeError, ValueError):
    """输入数据不满足评估前置条件（形状、维度、有限性等）。"""


class PoseQualityError(MoveScopeError, ValueError):
    """视频可解码，但姿态检测质量不足以支撑评估。"""
