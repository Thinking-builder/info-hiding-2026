"""
颜色变换策略 — 可替换的量化商计算方式

论文 Algorithm 1 Step 6 对每个通道计算标准差商 qc = σ_target / σ_source，
然后量化到 0.1 步长的 7 位整数（范围 0.1–12.8）。

本模块提供若干替代的量化商计算策略，每个函数签名均为：

    compute_qcode(source_std, target_std) -> np.ndarray  # shape (3,), int

返回三通道的 qcode 数组（整数，1–128），对应 q = qcode / 10。

使用方法
--------
在 core.py 的 create_mosaic() 中，将：

    qcode = np.clip(np.rint(quotient * 10), 1, 128).astype(int)

替换为所需策略的调用，例如：

    from sfvmosaic.optimize.color_transform import compute_qcode_global
    qcode = compute_qcode_global(source_std, target_std)
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 论文原始策略（基准）
# ──────────────────────────────────────────────────────────────────────────────

def compute_qcode_paper(
    source_std: np.ndarray,
    target_std: np.ndarray,
    zero_fallback: float = 0.1,
) -> np.ndarray:
    """论文基准：逐通道独立计算 qc = σ_target / σ_source（公式 3）。

    零标准差时使用 zero_fallback（默认 0.1，论文允许的最小值）。
    """
    quotient = np.divide(
        target_std,
        source_std,
        out=np.full(3, zero_fallback),
        where=source_std > 1e-12,
    )
    return np.clip(np.rint(quotient * 10), 1, 128).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# 替代策略
# ──────────────────────────────────────────────────────────────────────────────

def compute_qcode_global(
    source_std: np.ndarray,
    target_std: np.ndarray,
    zero_fallback: float = 0.1,
) -> np.ndarray:
    """全局单一量化商：三通道共用同一个 q。

    取三通道标准差的均值之比作为全局 q，三通道使用相同的 qcode。
    可减少色调偏移，但降低了各通道的独立调整能力。
    """
    s_mean = source_std.mean()
    t_mean = target_std.mean()
    if s_mean < 1e-12:
        q = zero_fallback
    else:
        q = t_mean / s_mean
    qcode_val = int(np.clip(round(q * 10), 1, 128))
    return np.full(3, qcode_val, dtype=int)


def compute_qcode_clamped(
    source_std: np.ndarray,
    target_std: np.ndarray,
    q_min: float = 0.5,
    q_max: float = 5.0,
    zero_fallback: float = 0.1,
) -> np.ndarray:
    """限幅版本：将 q 限制在 [q_min, q_max] 范围内。

    默认范围 [0.5, 5.0] 避免极端拉伸或压缩，减少 overflow/underflow 像素数量，
    从而降低残差开销和恢复信息总位数（对 Fig. 8(b) 有影响）。
    """
    quotient = np.divide(
        target_std,
        source_std,
        out=np.full(3, zero_fallback),
        where=source_std > 1e-12,
    )
    quotient = np.clip(quotient, q_min, q_max)
    return np.clip(np.rint(quotient * 10), 1, 128).astype(int)


def compute_qcode_lab(
    source_block: np.ndarray,
    target_block: np.ndarray,
    zero_fallback: float = 0.1,
) -> np.ndarray:
    """基于 LAB 色彩空间的量化商（Reinhard et al. 方法启发）。

    论文 Section III.A 提到本方法受 Reinhard et al. [22] 启发，
    原始 Reinhard 方法在 LAB 色彩空间进行颜色迁移。
    本函数将方块转换到近似 LAB 空间后计算 q，以减少颜色通道相关性。

    注意：本函数接受完整方块（而非预计算的标准差），签名与其他函数略有不同。
    在 core.py 中使用时需调整调用方式。
    """
    def _rgb_to_lab_approx(block: np.ndarray) -> np.ndarray:
        """近似 RGB→LAB 转换（线性化 + PCA 旋转近似）。"""
        b = block.astype(np.float64) / 255.0
        # 近似 sRGB 线性化
        b = np.where(b <= 0.04045, b / 12.92, ((b + 0.055) / 1.055) ** 2.4)
        # 近似 LAB 主轴（Ruderman et al. 对数变换近似）
        b = np.maximum(b, 1e-6)
        l = (np.log(b[:, :, 0]) + np.log(b[:, :, 1]) + np.log(b[:, :, 2])) / np.sqrt(3)
        a = (np.log(b[:, :, 0]) + np.log(b[:, :, 1]) - 2 * np.log(b[:, :, 2])) / np.sqrt(6)
        beta = (np.log(b[:, :, 0]) - np.log(b[:, :, 1])) / np.sqrt(2)
        return np.stack([l, a, beta], axis=2)

    src_lab = _rgb_to_lab_approx(source_block)
    tgt_lab = _rgb_to_lab_approx(target_block)
    src_std = src_lab.std(axis=(0, 1))
    tgt_std = tgt_lab.std(axis=(0, 1))
    quotient = np.divide(
        tgt_std, src_std, out=np.full(3, zero_fallback), where=src_std > 1e-12
    )
    return np.clip(np.rint(quotient * 10), 1, 128).astype(int)
