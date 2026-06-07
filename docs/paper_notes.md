# 论文算法要点与复现对照

> Lee, Y.-L. and Tsai, W.-H., "A New Secure Image Transmission Technique via Secret-Fragment-Visible Mosaic Images by Nearly-reversible Color Transformations," IEEE TCSVT, 2012.

---

## 一、研究目标

将秘密图像 $S$（大尺寸）转换为与任意预选目标图像 $T$ 外观相似、尺寸完全相同的
**secret-fragment-visible mosaic**，使马赛克可用作 $S$ 的视觉伪装。与传统图像加密
（输出噪声图）和数据隐藏（要求事先大幅压缩）不同，本方法无需图像数据库，且马赛克
本身即是有意义的图像。

---

## 二、马赛克创建算法（Algorithm 1）

### Stage 1：方块匹配（Steps 1–4）

**输入**：秘密图 $S$（$W_S \times H_S$）、目标图 $T$（调整为与 $S$ 同尺寸）、密钥 $K$。

1. 将 $S$ 和 $T$ 各划分为 $n$ 个大小为 $N_T \times N_T$ 的方块，分别组成集合
   $S_{\text{tile}} = \{T_1, \dots, T_n\}$ 和 $S_{\text{target}} = \{B_1, \dots, B_n\}$。

   其中 $n$ 满足 $m = \lceil \log_2 n \rceil$，$m$ 即索引所需位数。

2. 对每个方块，按公式（1）（2）计算三通道均值 $\mu_c$、标准差 $\sigma_c$，
   再取三通道平均标准差作为排序键：
   $$\bar{\sigma} = \frac{1}{3}(\sigma_R + \sigma_G + \sigma_B)$$

3. 将 $S_{\text{tile}}$ 和 $S_{\text{target}}$ 各自按 $\bar{\sigma}$ 升序排列后，
   按序位一一映射：$T_1 \to B_{j_1},\ T_2 \to B_{j_2},\ \dots,\ T_n \to B_{j_n}$。

4. 将秘密块 $T_i$ 按映射放入目标块 $B_{j_i}$ 所在位置，后续步骤在此基础上进行颜色变换。

> **复现细节**：标准差相同时以原始块索引为第二排序键，保证稳定排序。

---

### Stage 2：颜色变换（Steps 5–7）

对 $T_i$ 的每个像素 $p_k$，按论文公式（3）执行颜色变换：

$$c''_k = q_c \cdot (c_k - \mu_c) + \mu'_c \tag{3}$$

其中 $q_c = \sigma'_c / \sigma_c$ 为标准差商（$c = r, g, b$），变换后 $T'$ 的均值和方差与
$B_{j_i}$ 相同。

**量化规则**（保证嵌入比特数固定）：
- 均值 $\mu_c$、$\mu'_c$：舍入为整数，各 8 位，共 48 位（$2 \times 3 \times 8$）。
- 商 $q_c$：舍入到 0.1 步长，范围 $[0.1, 12.8]$，共 128 个值，7 位（$3 \times 7 = 21$ 位）。
  - 不允许 $q_c = 0$（使逆变换分母为零）；零标准差时取最小值 $0.1$。

**Overflow/Underflow 处理**（论文 Section III.C，公式 5）：

变换结果截断为整数后若 $\geq 255$ 或 $\leq 0$，先截到边界，再记录残差用于恢复：

$$c_S = \left\lceil \frac{255 - \mu'_c}{q_c} + \mu_c \right\rceil, \quad
  c_L = \left\lfloor \frac{0 - \mu'_c}{q_c} + \mu_c \right\rfloor \tag{5}$$

- **Overflow 像素**（变换后截断值 $\geq 255$）：$r_k = |c_k - c_S|$
- **Underflow 像素**（变换后截断值 $\leq 0$）：$r_k = |c_L - c_k|$
- **恰好落在 255/0 边界但未真正越界**的像素：也需记录残差（值为 0），
  以便恢复时与真正 overflow/underflow 像素区分。
- **非边界像素**：不记录残差，接受 $\leq 1$ 的量化/截断误差（论文称"nearly lossless"）。

残差值域为 $[0, 255]$（可验证），使用全局 Huffman 表 $HT$ 压缩编码（Step 9）。

---

### Stage 3：旋转优化（Step 8）

对颜色变换后的 $T'_i$，测试 $\{0°, 90°, 180°, 270°\}$ 四个旋转方向，
选取与 $B_{j_i}$ RMSE 最小的旋转角 $\theta_i$，将旋转后的 $T'_i$ 放入马赛克 $F$。

---

### Stage 4：恢复信息嵌入（Steps 9–12）

**Step 9**：统计全部方块的残差频率，构建全局 Huffman 表 $HT$。

**Step 10**：为每个方块 $T_i$ 构造比特流 $M_i$（论文定义格式）：

$$M_i = \underbrace{t_1 \cdots t_m}_{m \text{ 位}} \underbrace{r_1 r_2}_{2 \text{ 位}} \underbrace{m_1 \cdots m_{48}}_{48 \text{ 位}} \underbrace{q_1 \cdots q_{21}}_{21 \text{ 位}} \underbrace{d_1 \cdots d_k}_{\text{Huffman}}$$

| 字段 | 宽度 | 含义 |
|---|---|---|
| $t$ | $m = \lceil \log_2 n \rceil$ 位 | 目标块索引 $j_i$ |
| $r$ | 2 位 | 最优旋转角 $\theta_i$（0/1/2/3 对应 0°/90°/180°/270°）|
| $m$ | 48 位 | $T_i$ 和 $B_{j_i}$ 的三通道均值（各 8 位）|
| $q$ | 21 位 | 三通道标准差商 $q_c$（各 7 位，存储 $\text{qcode} - 1$）|
| $d$ | 变长 | 用 $HT$ 编码的残差（overflow 先，underflow 后，按通道 R/G/B 顺序）|

**Step 11**：将所有 $M_i$ 拼接得 $M_t$，用密钥 $K$ 加密得 $M'_t$，
使用 RCM 方案将 $M'_t$ 嵌入马赛克 $F$。

**Step 12**：构造全局信息流 $I$（包含 $N_i$：迭代次数；$N_{pair}$：最后一次迭代使用的像素对数；$HT$：Huffman 码长表），同样用 RCM 嵌入。

---

## 三、秘密图像恢复算法（Algorithm 2）

**输入**：马赛克 $F$、密钥 $K$。

### Stage 1：提取信息（Steps 1–5）

1. 从 $F$ 中逐层提取 $I$（外层）和 $M'_t$（内层），逆 RCM 恢复马赛克像素。
2. 用 $K$ 解密 $M'_t$ 得 $M_t$，用 $I$ 中的 $HT$ 解码残差。
3. 将 $M_t$ 分解为 $n$ 个比特流 $M_1, \dots, M_n$，解码各块的
   $j_i$、$\theta_i$、均值、量化商和残差序列。

### Stage 2：重建秘密图像（Steps 6–7）

对每个块 $T_i$（按秘密块顺序 $i = 1, \dots, n$）：

1. 从 $F$ 中取 $B_{j_i}$ 位置的块，按 $-\theta_i$ 逆旋转，得初始 $T'_i$。
2. 用逆变换（公式 4）恢复初始像素值：
   $$c_k = \frac{1}{q_c}(c''_k - \mu'_c) + \mu_c \tag{4}$$
3. 用恢复的 $\mu_c$、$\mu'_c$、$q_c$ 重算 $c_S$ 和 $c_L$（公式 5）。
4. 扫描 $T'_i$ 中值为 255 的像素（overflow/boundary）：$c_k = c_S + r_k$。
5. 扫描 $T'_i$ 中值为 0 的像素（underflow/boundary）：$c_k = c_L - r_k$。
6. 输出修正后的 $T_i$，clip 到 $[0, 255]$。

7. 将所有 $T_i$ 按原始位置拼合，输出秘密图像 $S$。

---

## 四、度量定义

| 度量 | 定义 | 用于 |
|---|---|---|
| RMSE | $\sqrt{\frac{1}{N}\sum(a_i - b_i)^2}$（全 RGB 通道） | 马赛克质量、恢复质量 |
| MSSIM | 按方块尺寸为窗口，逐通道计算 SSIM 后取平均（论文 [27]）| 马赛克结构相似度 |
| 恢复信息位数 | $|M_t| + |I|$ | 存储开销，对应 Fig. 8(b) |

---

## 五、与本复现的对照

### 完全对应

- 颜色变换公式（3）（4），量化规则（8 位均值，7 位商）
- $c_S/c_L$ 残差方案（公式 5），仅边界像素记录残差
- 按平均标准差排序匹配
- 四方向旋转选最小 RMSE
- Huffman 编码全局残差
- RCM 可逆嵌入（公式 6/7）
- RMSE 和 MSSIM（窗口 = 方块尺寸）
- $M_t$ 紧凑比特流格式（$m + 2 + 48 + 21$ bits/块 + 变长 Huffman）
- $I$ 信息流含 $N_i$、$N_{pair}$、$HT$
- 实验图像尺寸（1024×768 / 768×1024）、方块尺寸（8/16/32）、132 对实验

### 工程补全（论文未规定）

| 方面 | 本实现的选择 | 说明 |
|---|---|---|
| 加密算法 | AES-GCM + scrypt 密钥派生 | 论文仅说"用密钥加密" |
| 零标准差 | $q_c = 0.1$（最小允许值） | 论文仅说"不允许 $q_c=0$" |
| 排序稳定性 | 相同 $\bar{\sigma}$ 时以原始块索引为第二键 | 论文未规定 |
| 嵌入顺序 | 先嵌 $M'_t$（单层 RCM），再嵌 $I$（单层 RCM） | 论文描述两者均用 RCM 嵌入 |
| 容量不足 | paper 模式单层容量不足时报错 | 论文未讨论容量不足场景 |
| 像素配对 | R/G/B 通道分别按光栅顺序配对 | 论文未规定 RCM 配对方式 |

### 主要算法差异（已修复）

原实现对**所有像素**存储 zigzag 有符号校正，实现了完全无损恢复（RMSE = 0），
与论文"nearly lossless"不符。当前实现已改为论文的 $c_S/c_L$ 边界残差方案：
仅对 overflow/underflow 边界像素存储残差，非边界像素接受量化误差，
实验恢复 RMSE 约 0.4–0.9，符合论文 Fig. 8(c) 的量级。
