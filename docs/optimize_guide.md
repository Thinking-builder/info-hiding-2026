# 算法优化扩展指南

本文档说明如何在 `src/sfvmosaic/optimize/` 中添加新的优化策略，以及如何将其接入
论文核心流程。

---

## 一、整体架构与扩展点

论文的 mosaic 创建流程（`core.py: create_mosaic`）由以下几个关键步骤组成，
每个步骤都是可以替换或改进的**扩展点**：

```text
输入：秘密图 S、目标图 T、方块尺寸 block
  │
  ▼
[Step 1] 划分方块
  │        → 调整点：方块划分方式（当前：规则网格）
  ▼
[Step 2-3] 排序与匹配  ←──── block_match.py
  │        → 调整点：排序键函数（当前：三通道平均标准差）
  ▼
[Step 5-6] 颜色变换参数计算  ←──── color_transform.py
  │        → 调整点：量化商 qc 的计算方式（当前：逐通道 σ_target/σ_source）
  ▼
[Step 7] 像素变换 + overflow/underflow 处理
  │        → 调整点：截断方式（当前：np.trunc + clip）
  ▼
[Step 8] 旋转优化
  │        → 调整点：旋转度数（当前：0/90/180/270°）、评价指标（当前：RMSE）
  ▼
[Step 9-12] 残差编码 + RCM 嵌入
             → 调整点：Huffman 编码策略、RCM 嵌入顺序
```

---

## 二、替换分块匹配策略

### 现有策略（`optimize/block_match.py`）

| 函数 | 排序依据 | 适用场景 |
|---|---|---|
| `sort_key_avg_std` | 三通道平均标准差（论文基准） | 默认，效果已验证 |
| `sort_key_luminance_std` | 亮度通道标准差 | 对亮度匹配更敏感 |
| `sort_key_histogram_entropy` | 灰度直方图熵 | 纹理复杂度匹配 |
| `sort_key_max_channel_std` | 三通道标准差最大值 | 对饱和通道更敏感 |
| `sort_key_combined` | 标准差 + 熵的加权组合 | 可调权重 |

### 如何启用

在 `core.py` 的 `create_mosaic()` 函数中（约第 326–333 行），将：

```python
secret_order = sorted(
    range(len(secret_blocks)),
    key=lambda i: (float(secret_blocks[i].std(axis=(0, 1)).mean()), i),
)
target_order = sorted(
    range(len(target_blocks)),
    key=lambda i: (float(target_blocks[i].std(axis=(0, 1)).mean()), i),
)
```

替换为：

```python
from sfvmosaic.optimize.block_match import sort_key_luminance_std  # 选择策略

secret_order = sorted(
    range(len(secret_blocks)),
    key=lambda i: (sort_key_luminance_std(secret_blocks[i]), i),
)
target_order = sorted(
    range(len(target_blocks)),
    key=lambda i: (sort_key_luminance_std(target_blocks[i]), i),
)
```

### 添加新策略

在 `block_match.py` 中新增函数，签名为：

```python
def sort_key_my_method(block: np.ndarray) -> float:
    """说明：返回值越大，在排序中排名越靠后。"""
    ...
    return float(result)
```

---

## 三、替换颜色变换策略

### 现有策略（`optimize/color_transform.py`）

| 函数 | 说明 |
|---|---|
| `compute_qcode_paper` | 论文基准：逐通道 σ_target/σ_source |
| `compute_qcode_global` | 三通道共用同一全局 q |
| `compute_qcode_clamped` | 限制 q 范围，减少 overflow/underflow |
| `compute_qcode_lab` | LAB 色彩空间下计算 q（Reinhard 启发）|

### 如何启用

在 `core.py` 的 `create_mosaic()` 中（约第 348–355 行），将：

```python
quotient = np.divide(
    target_std, source_std,
    out=np.full(3, 0.1),
    where=source_std > 1e-12,
)
qcode = np.clip(np.rint(quotient * 10), 1, 128).astype(int)
```

替换为：

```python
from sfvmosaic.optimize.color_transform import compute_qcode_clamped

qcode = compute_qcode_clamped(source_std, target_std, q_min=0.5, q_max=5.0)
```

### 添加新策略

在 `color_transform.py` 中新增函数，签名为：

```python
def compute_qcode_my_method(
    source_std: np.ndarray,  # shape (3,), float64
    target_std: np.ndarray,  # shape (3,), float64
) -> np.ndarray:             # shape (3,), int, 范围 [1, 128]
    ...
    return qcode  # 对应 q = qcode / 10, 即 0.1–12.8
```

---

## 四、其他扩展点（需在 core.py 中直接修改）

### 4.1 旋转策略

当前实现测试 4 个旋转（0°/90°/180°/270°），评价指标为 RMSE。
可探索：

- 评价指标换为 SSIM（可能更接近视觉感知）
- 增加更多旋转角度（需先将块 resize 为方形）
- 对特定内容的块跳过旋转搜索以加速

相关代码位于 `core.py` 第 373–378 行：

```python
for rotation in range(4):
    candidate = np.rot90(converted, rotation)
    error = float(np.sqrt(np.mean((candidate.astype(float) - destination) ** 2)))
    if best_rmse is None or error < best_rmse:
        best_rmse, best_rotation, best_tile = error, rotation, candidate
```

### 4.2 残差编码

当前 paper 模式只对 overflow/underflow 边界像素记录残差（`_paper_residual_symbols`）。
可探索：

- 对 RMSE 贡献最大的 top-k 像素记录残差（非全边界）
- 使用算术编码代替 Huffman 编码以提高压缩率
- 分通道使用不同 Huffman 表

相关代码位于 `core.py` 第 104–128 行。

### 4.3 方块划分

当前使用规则网格（等尺寸无重叠方块）。可探索：

- 基于图像内容的自适应划分（如边缘感知分块）
- 非方形方块

---

## 五、运行对比实验

修改策略后，使用以下命令对比效果：

```bash
# 单图对趋势实验（快速验证）
uv run sfvmosaic experiment --blocks 8,16,32 --mode paper

# 132 对全量实验（完整评估）
uv run sfvmosaic experiment --blocks 8,16,32 --mode paper --pairs-dataset data/kodak

# 查看指标
cat outputs/experiment/trends.csv
cat outputs/experiment/pairs_summary.json
```

重点关注的指标：

- `core_mosaic_rmse`：马赛克与目标图的 RMSE（越低越好，体现视觉伪装质量）
- `paper_mt_bits`：Mt 的实际比特数（越低越好，体现存储效率）
- `recovered_rmse`：恢复秘密图的 RMSE（越低越好，体现近无损性）
- `core_mssim`：马赛克与目标图的结构相似度（越高越好）

---

## 六、代码风格约定

- 优化函数使用纯 numpy，不引入额外依赖
- 每个函数须有 docstring 说明算法原理和与论文基准的差异
- 新函数请在 `optimize/__init__.py` 的 `__all__` 中注册
- 若修改 `core.py`，请勿删除原有代码，改用条件开关或参数控制
