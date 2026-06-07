# Secret-Fragment-Visible Mosaic Images — Reproduction

> **Paper**: Ya-Lin Lee and Wen-Hsiang Tsai, "A New Secure Image Transmission Technique via Secret-Fragment-Visible Mosaic Images by Nearly-reversible Color Transformations," *IEEE Transactions on Circuits and Systems for Video Technology*, 2012.

本仓库对上述论文进行完整复现，实现了马赛克图像创建（hiding）与秘密图像恢复（recovery）的完整流程，并对论文第 V 节的实验结果进行了定量对照。

---

## 方法概述

该方法将一幅秘密图像 $S$ 转换为与任意目标图像 $T$ 外观相似、尺寸相同的马赛克图（mosaic），使马赛克在视觉上伪装成目标图像，同时允许持有密钥的接收方近无损地恢复原始秘密图像。

**核心思路**（见论文 Algorithm 1/2）：

1. 将 $S$ 和 $T$ 均划分为 $n$ 个大小相同的方块，按三通道平均标准差排序后一一匹配。
2. 对秘密块 $T_i$ 执行颜色变换 $c'' = q_c(c - \mu_c) + \mu'_c$，使其均值和方差与目标块 $B_{j_i}$ 一致。
3. 在四个旋转方向中选取与 $B_{j_i}$ RMSE 最小的方向，将变换后的块放入马赛克。
4. 对 overflow/underflow 边界像素记录残差（$|c_i - c_S|$ 或 $|c_L - c_i|$，论文公式 5），连同块索引、旋转角和量化参数一起经 Huffman 编码后加密，以可逆对比映射（RCM）嵌入马赛克。
5. 恢复时：提取并解密嵌入信息，逆变换 $c = (c'' - \mu'_c)/q_c + \mu_c$，对边界像素加回残差，得到近无损的秘密图像。

---

## 仓库结构

```text
.
├── src/sfvmosaic/
│   ├── core.py        # 核心算法：颜色变换、残差计算、紧凑比特流打包/解析
│   ├── rcm.py         # 可逆对比映射（RCM）嵌入与提取
│   ├── huffman.py     # 规范 Huffman 编解码
│   ├── api.py         # 公开接口：hide / reveal / core_roundtrip
│   ├── metrics.py     # RMSE 和 MSSIM 度量
│   ├── experiment.py  # 趋势实验与 132 对实验
│   ├── io.py          # 图像读写与尺寸规范化
│   ├── crypto.py      # AES-GCM 加密（工程补全）
│   ├── cli.py         # 命令行入口
│   ├── collect.py     # Kodak 数据集下载
│   └── optimize/      # 可插拔算法优化模块（面向扩展）
│       ├── block_match.py      # 可替换的分块匹配策略
│       └── color_transform.py  # 可替换的颜色变换策略
├── tests/             # 18 项单元测试
├── docs/
│   ├── paper_notes.md         # 论文算法要点与复现对照
│   ├── reproduction_report.md # 实验结果报告
│   └── optimize_guide.md      # 算法优化扩展指南
├── configs/default.json
├── data/kodak/        # 实验数据集（12 张 Kodak 图像）
├── outputs/           # 运行结果（demo 和 experiment）
├── reference.pdf      # 原论文
├── secret.png         # 演示用秘密图像
└── target.png         # 演示用目标图像
```

---

## 安装

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
# 克隆仓库后安装依赖
uv sync

# 或使用 pip
pip install -e ".[dev]"
```

**运行环境**：Python ≥ 3.10，依赖 `numpy`、`pillow`、`matplotlib`、`cryptography`（测试还需 `scikit-image`）。

---

## 快速开始

### 演示（一键运行完整流程）

```bash
uv run sfvmosaic demo --mode paper
```

输出保存在 `outputs/demo/`，包括核心马赛克、嵌入后马赛克、恢复图像和 `metrics.json`。

### 手动隐藏与提取

```bash
# 隐藏秘密图像
uv run sfvmosaic hide secret.png target.png outputs/mosaic.png \
    --block 8 --key "my-secret-key" --mode paper

# 从马赛克恢复秘密图像
uv run sfvmosaic extract outputs/mosaic.png outputs/recovered.png \
    --key "my-secret-key"
```

### 复现 Fig. 8 实验

```bash
# 下载 Kodak 数据集（约 10 MB）
uv run sfvmosaic collect-data

# 单图像对趋势实验（对应论文 Fig. 8 各曲线）
uv run sfvmosaic experiment --blocks 8,16,32 --mode paper --embedded

# 全部 132 对实验（对应论文 Fig. 8 橙色均值曲线）
uv run sfvmosaic experiment --blocks 8,16,32 --mode paper \
    --pairs-dataset data/kodak
```

实验结果输出至 `outputs/experiment/`，包含 CSV 数据和趋势图 `fig8_trends.png`。

### 运行测试

```bash
uv run pytest -v
```

---

## 实验结果

### Demo（222×296，方块 37×37，paper 模式）

| 指标 | 数值 |
|---|---:|
| 核心马赛克 RMSE（vs 目标图） | 45.61 |
| 核心马赛克 MSSIM（vs 目标图） | 0.255 |
| 嵌入后马赛克 RMSE（vs 目标图） | 48.95 |
| 恢复秘密图 RMSE（vs 原始） | 0.363 |
| 论文协议恢复信息（$M_t + I$） | 29,380 bits |

### Fig. 8 趋势（768×1024，paper 模式）

对应论文 Fig. 8(a)(b)(c)(d)，使用固定尺寸 768×1024 图像：

| 方块尺寸 | 马赛克 RMSE | MSSIM | 恢复 RMSE | 总恢复位数（$M_t + I$） |
|---|---:|---:|---:|---:|
| 8×8   |  9.86 | 0.711 | 0.411 |   1,095,842 |
| 16×16 | 18.05 | 0.507 | 0.367 |     343,289 |
| 32×32 | 27.80 | 0.342 | 0.326 |     197,275 |

**趋势与论文一致**：方块越小，马赛克越接近目标图（RMSE 低、MSSIM 高），但需要更多比特存储恢复信息；恢复 RMSE 随方块减小而略微增大，符合论文 Fig. 8(c) 的走势。

### 132 对实验（Kodak 数据集，paper 模式）

12 张 Kodak 图像的全部 $12 \times 11 = 132$ 有序图像对，对应论文 Fig. 8 橙色均值曲线：

| 方块尺寸 | 均值马赛克 RMSE | 均值 MSSIM | 均值恢复 RMSE | RMSE ≥ 1 的对数 | 均值总恢复位数 |
|---|---:|---:|---:|---:|---:|
| 8×8   | 18.76 | 0.615 | 0.819 | 37/132 |   1,136,721 |
| 16×16 | 25.03 | 0.488 | 0.848 | 38/132 |     392,624 |
| 32×32 | 30.56 | 0.385 | 0.823 | 32/132 |     254,849 |

完整分布统计（min/std/P50/P95/max）见 `outputs/experiment/pairs_summary.json`。

---

## 实现说明

### 与论文严格对应的部分

| 论文描述 | 本实现 | 位置 |
|---|---|---|
| 颜色变换 $c'' = q_c(c-\mu_c)+\mu'_c$（公式 3/4） | 完全对应 | `core.py` |
| $q_c$ 量化为 7 位（0.1–12.8，步长 0.1） | `clip(rint(q×10), 1, 128) / 10` | `core.py` |
| 均值量化为 8 位整数 | `np.rint(...).astype(int)` | `core.py` |
| $c_S/c_L$ 残差（公式 5），仅边界像素 | `ceil/floor + \|c_i - c_S\|` | `core.py` |
| 按平均标准差排序，一一匹配 | `std(axis=(0,1)).mean()` 排序 | `core.py` |
| 四方向旋转选最小 RMSE | `np.rot90` × 4 | `core.py` |
| Huffman 编码残差 | 规范 Huffman | `huffman.py` |
| RCM 可逆嵌入（公式 6/7） | 完全对应 | `rcm.py` |
| RMSE 和 MSSIM（窗口 = 方块尺寸） | 已通过 scikit-image 交叉验证 | `metrics.py` |
| 图像尺寸 1024×768 / 768×1024 | `resize_canonical` | `io.py` |
| 方块尺寸 8/16/32，132 对实验 | 完整实现 | `experiment.py` |
| $M_t$ 紧凑比特流格式：$m + 2 + 48 + 21$ bits/块 | `_BitWriter/_pack_paper_metadata` | `core.py` |
| $I$ 信息流含 $N_i$、$N_{pair}$、$HT$ | `_pack_paper_info` | `core.py` |

### 工程补全（论文未规定的部分）

| 项目 | 本实现的选择 |
|---|---|
| 加密算法 | AES-GCM，密钥由 scrypt 派生（论文仅说"用密钥加密"）|
| 零标准差处理 | $q_c$ 设为最小值 0.1（论文仅说"不允许 $q_c = 0$"）|
| 排序稳定性 | 标准差相同时按原始块索引为第二键 |
| 嵌入架构 | paper 模式：两次单层 RCM（先嵌 $M_t'$，再嵌 $I$）；robust 模式：多层 RCM |
| 容量不足 | paper 模式单层容量不足时直接报错；robust 模式自动多层嵌入 |

### 两种运行模式

- **`paper` 模式**（推荐用于与论文对比）：使用论文的 $c_S/c_L$ 残差方案（仅边界像素），生成紧凑二进制 $M_t$ 和独立 $I$ 信息流，论文协议的 `nearly lossless` 恢复（非边界像素有 ≤1 的量化误差）。
- **`robust` 模式**：直接保存边界像素的原始值，配合多层 RCM 嵌入，消除 $c_S/c_L$ 歧义，工程可靠性更高。

---

## 数据集

实验使用 [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/)（12 张图像）作为论文原始数据集的公开替代（论文原始图像来自作者机构网站，需原论文参考文献 [32]）。

下载方式：

```bash
uv run sfvmosaic collect-data
# 图像保存至 data/kodak/，含 SHA256 校验清单
```

---

## 引用

```bibtex
@article{lee2012secret,
  title   = {A New Secure Image Transmission Technique via
             Secret-Fragment-Visible Mosaic Images by
             Nearly-Reversible Color Transformations},
  author  = {Lee, Ya-Lin and Tsai, Wen-Hsiang},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2012}
}
```

---

## 许可

本仓库仅用于学术研究与课程复现目的。
