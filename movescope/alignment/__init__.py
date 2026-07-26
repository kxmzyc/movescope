"""DTW 时序对齐子包。

对外接口与旧的单文件 movescope/alignment.py 保持一致：
DTWAligner 与 WeightedSegmentedDTWAligner 仍从 movescope.alignment 导入。
"""

from movescope.alignment.aligners import DTWAligner, WeightedSegmentedDTWAligner
from movescope.alignment.dtw import AlignmentPath

__all__ = ["AlignmentPath", "DTWAligner", "WeightedSegmentedDTWAligner"]
