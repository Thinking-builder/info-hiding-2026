"""
sfvmosaic.optimize — 可插拔算法优化模块

本包提供论文核心流程中各关键步骤的替代实现，供实验性扩展使用。
每个模块都与 core.py 中对应步骤的接口兼容，可通过修改 core.py
中的调用点来启用。

扩展点概览：
  block_match.py      - 分块匹配策略（排序键函数）
  color_transform.py  - 颜色变换参数计算方式

详细使用说明：docs/optimize_guide.md
"""

from .block_match import sort_key_avg_std, sort_key_luminance_std, sort_key_histogram_entropy
from .color_transform import compute_qcode_paper, compute_qcode_global

__all__ = [
    "sort_key_avg_std",
    "sort_key_luminance_std",
    "sort_key_histogram_entropy",
    "compute_qcode_paper",
    "compute_qcode_global",
]
