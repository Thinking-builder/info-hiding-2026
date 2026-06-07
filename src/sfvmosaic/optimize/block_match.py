"""
分块匹配策略 — 可替换的排序键函数

论文 Algorithm 1 Step 3 要求按方块的某种特征值对秘密块集合和目标块集合
分别排序，然后按排名一一映射。论文使用三通道平均标准差作为排序键。

本模块提供若干替代策略，每个函数的签名均为：

    sort_key(block: np.ndarray) -> float

其中 block 是形状为 (H, W, 3) 的 uint8 数组，返回值越大排名越靠后。

使用方法
--------
在 core.py 的 create_mosaic() 中，将：

    key=lambda i: (float(secret_blocks[i].std(axis=(0, 1)).mean()), i)

中的 std(...).mean() 部分替换为所需策略，例如：

    from sfvmosaic.optimize.block_match import sort_key_luminance_std
    key=lambda i: (sort_key_luminance_std(secret_blocks[i]), i)
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 论文原始策略（基准）
# ──────────────────────────────────────────────────────────────────────────────

def sort_key_avg_std(block: np.ndarray) -> float:
    """论文基准：三通道标准差的均值（Algorithm 1 Step 2-3）。"""
    return float(block.std(axis=(0, 1)).mean())


# ──────────────────────────────────────────────────────────────────────────────
# 替代策略
# ──────────────────────────────────────────────────────────────────────────────

def sort_key_luminance_std(block: np.ndarray) -> float:
    """亮度通道标准差。

    将 RGB 转换为感知亮度 Y = 0.299R + 0.587G + 0.114B，
    仅用亮度的标准差排序。对灰度感知更敏感，忽略色调差异。
    """
    b = block.astype(np.float64)
    luminance = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    return float(luminance.std())


def sort_key_histogram_entropy(block: np.ndarray) -> float:
    """灰度直方图熵。

    计算方块灰度图的 Shannon 熵，熵越高表示方块内容越复杂。
    可使匹配时"纹理丰富"的秘密块与"纹理丰富"的目标块配对。
    """
    gray = block.mean(axis=2).astype(np.uint8).reshape(-1)
    counts = np.bincount(gray, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def sort_key_max_channel_std(block: np.ndarray) -> float:
    """三通道标准差的最大值（而非均值）。

    对单通道饱和（如纯红色高对比块）更敏感。
    """
    return float(block.std(axis=(0, 1)).max())


def sort_key_combined(block: np.ndarray, w_std: float = 0.7, w_entropy: float = 0.3) -> float:
    """加权组合策略（平均标准差 + 亮度熵）。

    可通过调整 w_std 和 w_entropy 平衡两种特征的权重。
    """
    std_val = sort_key_avg_std(block) / 128.0          # 归一化到约 [0, 1]
    ent_val = sort_key_histogram_entropy(block) / 8.0  # 最大熵为 8 bits
    return float(w_std * std_val + w_entropy * ent_val)
