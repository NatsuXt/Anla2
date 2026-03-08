# Scatter Code 局部最优的数学分析

**背景**: Config B (V=256, D=64, V/D=4) 在 v5（PhaseTwist）和 v6（HolomorphicActivation）两版实验中均收敛至相同的"高维散点编码"，准确率 99.73% 但环拓扑完全丧失（NN% ≈ 2%）。v6 替换激活函数后振荡未消除、拓扑未恢复，说明问题根源不在激活函数而在架构与损失函数的结构性缺陷。

---

## 一、两类解的数学刻画

当前任务要求在 \(\mathbb{C}^D\)（\(D=64\)）中安放 \(V=256\) 个 embedding 向量 \(\{w_k\}_{k=0}^{V-1}\)，使得 Boltzmann 分类损失最小化。从数学上看，至少存在两类本质不同的"好解"：

### 解 A：高维散点码（Scatter Code）

所有 token 对之间的距离近似相等：

\[
\|w_i - w_j\|_2 \approx c, \quad \forall\, i \neq j
\]

这等价于 256 个点近似构成 \(\mathbb{R}^{128}\)（复数 64 维 = 实数 128 维）中的**正则单纯形**顶点。Johnson-Lindenstrauss 引理保证：当 \(D_{\text{real}} = 128 \gg \ln V = \ln 256 \approx 5.5\) 时，128 维空间有充裕的容量容纳 256 个近等距点。

v5/v6 的实验数据完美符合这个描述：距离分布近似正态（median = 10.73, std ≈ 0.8），距离矩阵均匀无结构，PCA 方差完全分散。

### 解 B：频率编码（Frequency Code，保拓扑解）

每个维度编码一个频率分量：

\[
w_k^{(d)} = r_d \cdot \exp\!\Big(i \cdot \frac{2\pi f_d \cdot k}{V}\Big), \quad f_d \in \{1, 2, \ldots\}
\]

此时 token 对 \((i, j)\) 的距离成为环距的函数：

\[
\|w_i - w_j\|^2 = \sum_{d=1}^{D} 2r_d^2 \Big[1 - \cos\!\Big(\frac{2\pi f_d (i-j)}{V}\Big)\Big] = g\big(|i-j| \bmod V\big)
\]

这自动保持了环拓扑——环上近邻在 embedding 空间也是近邻。

---

## 二、为什么 Scatter Code 是"默认吸引子"

### 2.1 初始化天然偏向 Scatter Code

ComplexEmbedding 初始化时，\(V\) 个向量在 \(\mathbb{C}^D\) 中随机采样。高维概率论的经典结论是：当 \(D\) 足够大时，任意两个独立随机向量之间的距离**高度集中**在 \(\sqrt{2D} \cdot \sigma\) 附近。也就是说，**随机初始化本身就已经是一个近似 Scatter Code**。

优化器从这个初始点出发，只需要微调各 embedding 的位置使能量分离更清晰——不需要全局重组。从动力学角度看，Scatter Code 方案的吸引域（basin of attraction）**包含了几乎整个初始化空间**。

### 2.2 频率编码需要全局协调

相比之下，解 B 要求 256 个 embedding 同时满足一个**高度协调的全局约束**：所有维度的相位必须按照 \(\theta_d(k) = 2\pi f_d k / V + \phi_d\) 精确排列。这意味着每个 embedding 的每个维度都受到与其他所有 embedding 的联合约束。

从优化角度看，这相当于要求梯度下降同时协调 \(V \times D = 256 \times 64 = 16384\) 个复数参数走向一个高度特殊的子流形。这个子流形在参数空间中的"体积"远小于 Scatter Code 解集的体积。

### 2.3 信息论视角：过参数化允许"懒惰解"

分类 256 个 token 仅需要 \(\log_2 256 = 8\) 比特的信息。而每个 embedding 有 128 个实数自由度。这意味着系统有 \(128 / 8 = 16\) 倍的冗余度。在如此巨大的冗余下，存在指数级数量的有效分类方案。Scatter Code 是其中"最通用"的一类——它不对 embedding 施加任何结构性约束，只要求点对距离大于某个阈值。频率编码则是这个庞大解空间中一个零测度的特殊子集。

---

## 三、Boltzmann 损失的梯度结构——为什么它不推向拓扑解

### 3.1 Boltzmann 损失对 embedding 的梯度

对于 token \(n\) 的预测向量 \(z\)，Boltzmann 概率为：

\[
p_k = \frac{\exp(-E_k / \tau)}{\sum_{j} \exp(-E_j / \tau)}, \quad \tau = \mathrm{std}\{E_j : j \neq \mathrm{target}\}
\]

损失 \(L = -\log p_{\mathrm{tgt}}\) 对 embedding \(w_k\) 的梯度（通过 \(E_k\) 传递）具有以下结构：

- 对 **target embedding** \(w_{\mathrm{tgt}}\)：梯度方向是将 \(z\) **拉向** \(w_{\mathrm{tgt}}\)（减小 \(E_{\mathrm{tgt}}\)）
- 对 **竞争者 embedding** \(w_k\)（\(k \neq \mathrm{tgt}\)）：梯度方向是将 \(w_k\) **推离** \(z\)，强度正比于 \(p_k\)——即**最近的竞争者受到最强的排斥力**

关键观察：**这个梯度完全不包含任何关于 ring distance 的信息。** 排斥力只取决于 embedding 空间中的能量距离 \(E_k\)，而不取决于 token \(k\) 在环上与 target 的拓扑距离。

形式化地写：令 \(d_{\mathrm{ring}}(i,j) = \min(|i-j|, V-|i-j|)\) 为环距。Boltzmann 梯度中出现的是：

\[
\frac{\partial L}{\partial w_k} \propto p_k \cdot \frac{\partial E_k}{\partial w_k}
\]

其中 \(p_k\) 只取决于 \(\{E_j\}_{j \neq \mathrm{tgt}}\)，完全不包含 \(d_{\mathrm{ring}}(k, \mathrm{tgt})\)。**环拓扑在损失函数中是完全不可见的。**

### 3.2 "拓扑盲"梯度的后果

由于梯度只关心"谁是最近的能量竞争者"而不关心"谁是环上的近邻"，优化过程会将每个 token 推离其当前最近的 embedding 邻居，不管这个邻居在环上是近还是远。

在 Scatter Code 附近，所有竞争者距离近似相等 → 排斥力近似均匀 → 系统维持各向同性的均匀散布。这是一个**自洽的不动点**：均匀散布 → 均匀排斥 → 保持均匀散布。

要逃离这个不动点进入频率编码，需要一个"对称性破缺"力——将环上近邻 token 拉近而将环上远邻推远。但 Boltzmann 梯度中**不存在这样的力**。

### 3.3 Encoder-Decoder 放大了这个问题

v5/v6 引入的 Encoder-Decoder 架构进一步加剧了局部最优问题。Decoder 是一个可学习的复数线性映射 \(D_\theta: \mathbb{C}^D \to \mathbb{C}^D\)。这意味着即使 Transformer 的内部表示具有某种结构，Decoder 也可以通过一个任意线性变换将其映射到 Scatter Code 形式的 embedding 空间。

数学上，如果 Transformer 输出 \(h\) 落在某个子空间 \(\mathcal{M}\) 上，Decoder 可以学习一个线性映射 \(D_\theta\) 使得 \(\{D_\theta(h_k)\}\) 在 embedding 空间中呈 Scatter Code 分布。Decoder 的存在**解耦了 Transformer 内部表示的几何结构与 embedding 空间的几何结构**，使得保拓扑的梯度信号（即使存在）也无法穿透 Decoder 传递到 embedding 层。

---

## 四、Phase 3 振荡的动力学分析——平坦极小值上的随机游走

### 4.1 Scatter Code 极小值的 Hessian 结构

在 Scatter Code 极小值附近，考虑 Boltzmann loss 对 embedding 的 Hessian 矩阵。由于所有竞争者距离近似相等（\(E_k \approx \bar{E}\) for \(k \neq \mathrm{tgt}\)），Boltzmann 概率近似均匀：\(p_k \approx 1/(V-1)\)。

在这种"均匀排斥"条件下，Hessian 在 embedding 空间中具有大量**接近零的特征值**。直觉上：如果所有竞争者距离相等，那么在保持等距约束的超曲面上"滑动"一个 embedding 几乎不改变 loss。这意味着 Scatter Code 极小值不是一个尖锐的"点"，而是一个**高维平坦谷**。

### 4.2 振荡的数学解释

在平坦谷中，梯度向量几乎正交于谷底方向，但有微小的沿谷方向分量（来自 batch 采样噪声和 BERT masking 的随机性）。固定学习率的 SGD/Adam 在这种景观上表现为：

\[
w_{t+1} = w_t - \eta \cdot g_t, \quad g_t = \nabla L(w_t; \mathrm{batch}_t)
\]

其中 \(g_t\) 的"有用分量"（沿梯度下降方向）很小，而"噪声分量"（沿平坦谷方向）可能更大。结果是 \(w_t\) 在平坦谷中做**随机游走**，loss 随着随机游走的位置波动。

这完美解释了实验观察：

- 后期 loss 振荡系数 ~50%（**大噪声 / 小梯度**的特征）
- 换激活函数不改变振荡（**振荡来自景观结构而非前向传播**）
- L\_Elegant 持续缓慢下降（**在平坦谷中仍有微弱的梯度方向**）
- lr 衰减应能抑制振荡（**减小 \(\eta\) 直接压制随机游走步长**）

### 4.3 τ 的正反馈效应

将 Boltzmann 温度 \(\tau = \mathrm{std}(E_{\text{comp}})\) 纳入动力学分析后，情况更加复杂。考虑一个扰动链：

\[
\delta w \xrightarrow{\text{Decoder}} \delta \tilde{E}_k \xrightarrow{\text{std}} \delta \tau \xrightarrow{-E_k/\tau^2} \delta p_k \xrightarrow{} \delta L \xrightarrow{} \delta g
\]

当 embedding 在平坦谷中移动时，竞争者能量分布的标准差 \(\tau\) 也随之波动。\(\tau\) 的减小会使 softmax 变锐、梯度变大，可能导致下一步 overshoot；反之 \(\tau\) 增大会使梯度变小、步长变短。这构成了一个**延迟反馈振荡器**，其频率由 \(\eta / \tau^2\) 的量级决定。

实验中 \(\tau\) 后期波动幅度很小（~0.005），但由于 \(\tau \approx 0.19\) 已经很小，\(1/\tau^2 \approx 28\) 的放大因子使得即使微小的 \(\delta\tau\) 也能产生可观的 loss 变化。

---

## 五、为什么 HolomorphicActivation 没有帮助逃离局部最优

### 5.1 激活函数不改变损失函数的梯度结构

v5→v6 的核心修改是 FFN 中的激活函数。但如上所述，局部最优的根因在于 **Boltzmann 损失对环拓扑完全盲**。无论 FFN 的非线性是 PhaseTwist 还是 \(z + \alpha z^2\)，都不改变这个根本事实：loss 的梯度中不包含 \(d_{\mathrm{ring}}\)，因此不存在将模型推向频率编码的力。

### 5.2 二次谐波的频率创造为什么没有效果

理论上 \(z^2 = r^2 e^{2i\theta}\) 产生二次谐波，可以将 \(D=64\) 个基频扩展为更多频率组合。但这个能力的前提是**基频本身已经存在**——即 embedding 已经具有 \(\theta_d(k) = 2\pi f_d k / V\) 的结构。

在 Scatter Code 极小值处，embedding 的相位是完全随机的（Phase Linearity = 0）。对随机相位施加 \(z^2\)：

\[
(z^{(d)})^2 = r_d^2 \exp(2i\theta_d) \quad \text{where } \theta_d \sim \mathrm{Uniform}(-\pi, \pi)
\]

\(2\theta_d \bmod 2\pi\) 仍然是均匀分布——**对随机相位做倍频仍然得到随机相位**。二次谐波只有在基频已经有序的前提下才有"频率创造"的意义。

### 5.3 Cauchy-Riemann AM↔PM 耦合为什么没有效果

同理，Cauchy-Riemann 方程保证的 AM↔PM 耦合：

\[
\frac{\partial \Phi}{\partial r} = -\frac{1}{r}\frac{\partial \ln R}{\partial \theta}
\]

在 Scatter Code 处，模长和相位的联合分布没有结构（相位-模长散点图呈均匀散布）。AM↔PM 耦合在一个没有结构的信号上施加非线性，产生的仍然是没有结构的输出。**全纯非线性的优势需要在有序的信号上才能体现。**

---

## 六、逃离局部最优的数学条件

### 6.1 需要什么样的梯度信号？

要让优化过程发现频率编码解，需要在梯度中引入与环距相关的项。形式化地，需要一个辅助力：

\[
F_{\mathrm{topo}}(w_i, w_j) \propto -\nabla_{w_i} \Phi\big(d_{\mathrm{ring}}(i,j),\, \|w_i - w_j\|\big)
\]

其中 \(\Phi\) 是一个势函数，使得：

- 当 \(d_{\mathrm{ring}}(i,j)\) 小但 \(\|w_i - w_j\|\) 大时产生**吸引力**
- 当 \(d_{\mathrm{ring}}(i,j)\) 大但 \(\|w_i - w_j\|\) 小时产生**排斥力**

最简单的形式是应力函数（MDS stress）：

\[
L_{\mathrm{topo}} = \sum_{i < j} \Big(\|w_i - w_j\| - \lambda \cdot d_{\mathrm{ring}}(i,j)\Big)^2
\]

这直接在梯度中注入了 \(d_{\mathrm{ring}}\) 信息。

### 6.2 约束 Decoder 的数学意义

如果 Decoder 被约束为酉矩阵 \(U\)（\(U^\dagger U = I\)），则：

\[
\|D_\theta(h_i) - D_\theta(h_j)\| = \|U(h_i - h_j)\| = \|h_i - h_j\|
\]

酉变换**保持距离**。这意味着 Transformer 内部表示的几何结构必须与 embedding 空间一致——Transformer 不能再"偷懒"使用一个与 embedding 几何无关的内部表示，然后靠 Decoder 做任意映射来补救。

但仅靠酉约束仍然不能保证拓扑保持，因为 Boltzmann 梯度本身仍然是拓扑盲的。酉约束消除了 Decoder 的"捷径"，但不创造新的拓扑梯度信号。

### 6.3 V/D 压力能否强制拓扑涌现？

在更高的 \(V/D\) 比下，Scatter Code 的可行性会降低。粗略估计，\(V\) 个等距点在 \(\mathbb{R}^{2D}\) 中的最大数目约为 \(2D + 1\)。当 \(V > 2D + 1 = 129\) 时（Config B 中 \(V = 256 > 129\)），完美等距不可能实现，但"近似等距"在高维中仍然是宽松的。

只有当 \(V/D\) 足够大，使得 Scatter Code 无法提供足够的能量分离（即 \(E_{\mathrm{tgt}}\) 和 \(E_{\mathrm{nw}}\) 之间的 gap 太小以至于分类失败）时，模型才会被迫寻找更结构化的编码方案。但从 v4 的数据看，即使在 \(V/D = 4\) 时 Scatter Code 已经达到 99.7% 准确率——能量分离远远充足。可能需要 \(V/D \geq 16\) 甚至更高才能在能量层面迫使 Scatter Code 失效。

### 6.4 全局最优解的存在性与可达性

实际上，频率编码解是否真的是"全局最优"？

对于 Boltzmann 损失本身，Scatter Code 和频率编码在 \(p_{\mathrm{tgt}}\) 方面可能非常接近——两者都能将 target 能量推到极低值。频率编码在 **Boltzmann loss 意义下不一定优于 Scatter Code**。它的优势在于可解释性和拓扑保持——但这些不是 loss 的组成部分。

换言之：**当前 loss 函数下频率编码可能不是全局最优，而只是另一个等能量的局部最优**。如果我们想让模型选择频率编码，就必须修改 loss 使得频率编码在新 loss 下具有更低的能量。

---

## 七、总结：局部最优的结构

将以上分析综合，当前架构的局部最优具有以下数学结构：

1. **解的性质**：\(\mathbb{C}^{64}\) 中 256 个近等距散点（近似正则单纯形顶点）
2. **吸引域大小**：几乎覆盖整个参数空间（任何随机初始化都在此吸引域内）
3. **不动点自洽性**：均匀散布 → 均匀排斥 → 均匀散布（Boltzmann 梯度的自洽循环）
4. **Hessian 特征**：大量接近零的特征值，对应平坦谷中的等距变换方向
5. **与频率编码的势垒**：不是一个局部势垒（可以通过更大 lr 翻越），而是一个**信息势垒**——梯度中不包含到达频率编码所需的方向信息

**"局部最优"这个说法准确但需要注意**：这不是通常意义上可以通过模拟退火或学习率调整逃离的"窄谷"型局部最优，而是一个**广阔的平坦盆地**。逃离它不需要更大的步长，而需要**不同方向的梯度信号**——这只能通过修改损失函数或引入拓扑约束来获得。

---

## 八、可行方案优先级排序

基于以上数学分析，逃离 Scatter Code 局部最优的方案按有效性排序：

| 优先级 | 方案 | 原理 | 预期效果 |
|--------|------|------|----------|
| **1** | **拓扑正则化损失** | 在梯度中直接注入 \(d_{\mathrm{ring}}\) 信息 | 打破 Scatter Code 的自洽循环 |
| **2** | **约束 Decoder 为酉映射** | 消除 Decoder 的捷径学习路径 | 迫使 Transformer 内部保拓扑 |
| **3** | **后期 lr 衰减** | 减小平坦谷中的随机游走步长 | 抑制 Phase 3 振荡（但不逃离局部最优） |
| **4** | **增大 V/D 压力** | 迫使 Scatter Code 在容量上失效 | 可能自发涌现结构化编码 |
| **5** | **回退 span masking** | 恢复局部连续性的归纳偏置 | 辅助性，不解决根本问题 |

注意：方案 3 仅抑制振荡但不逃离局部最优；方案 1 和 2 才能真正改变优化景观的结构。
