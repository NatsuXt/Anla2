# Project Anla v5→v6 研究过程汇总

---

## 一、v5 实验数据分析（起点）

Config B (V=256, D=64, V/D=4) 在 v5 架构下取得了表面上优异的结果：训练准确率 100%，测试准确率 99.6%±0.53%。Boltzmann 热力学全面健康，τ 从 v4 的 ~10.8 稳定收敛至 ~0.21，neg% 彻底归零。

但数据中隐藏着两个深层问题：

**问题 1：后期训练的系统性振荡。** loss 在 epoch 8000 之后于 0.03–0.18 之间反复波动，无法持续下降。p_target 等指标表明模型已经很好地完成了拟合，优化器、warmup、学习率调度也保证了后期步长足够小，不应该产生这样的躁动。

**问题 2：环拓扑完全丧失。** NN% 从 v4 的 43.8% 降至 26.6%，PCA 投影中看不到环形结构，embedding 距离与 ring distance 几乎不相关。相位线性度为 0——没有任何维度呈现频率编码。模型选择了"高维散点编码"的捷径。

---

## 二、对 PhaseTwist 的反思

### 2.1 对后期振荡的根因追溯

> **⭐ 关键观点 1：PhaseTwist 是一个内嵌在前向传播中的扰动放大器。**

PhaseTwist 的耦合（\(\gamma r\) 和 \(\beta\cos(\theta-\varphi)\)）是**固定函数形式、全局标量参数、每次前向传播无条件施加**的。当模型已接近收敛时，任何微小的模长扰动 \(\delta r\) 都会被 \(\gamma r\) 项转化为相位偏移，相位偏移改变 Boltzmann 能量，产生新梯度修正模长，模长修正又通过 \(\gamma r\) 产生新的相位偏移——形成**耦合振荡器反馈环路**。3 层 block 叠加意味着这个耦合被施加 3 次，每次都放大扰动。

\(\gamma\) 和 \(\beta\) 是全局标量，对所有 256 个 token、所有 FFN 隐层维度施加相同强度的耦合，无法区分"需要强耦合"和"耦合会造成不稳定"的维度。网络没有能力选择性地关闭不需要的耦合通道。

### 2.2 AM↔PM 耦合是否需要 PhaseTwist 来提供？

初始判断是架构本身（ComplexLinear 的矢量合成、残差连接的复数加法、HolographicAttention 的干涉）已经天然具备 AM↔PM 耦合能力，PhaseTwist 是多余的。

但这个判断在后续被修正——

---

## 三、对复数乘法的纠错

> **⭐ 关键观点 2：复数乘法不提供 AM↔PM 耦合。**

这是对上一步结论的重要修正。复数乘法在极坐标下：

\[
$z_a \cdot z_b = (r_a \cdot r_b) \cdot e^{i(\theta_a + \theta_b)}$
\]

输出模长 \(r_a r_b\) **只取决于输入模长**（AM × AM → AM），输出相位 \(\theta_a + \theta_b\) **只取决于输入相位**（PM + PM → PM）。模长和相位在乘法中是**完全正交的两个通道**，互不干涉。

因此 ComplexLinear（本质是复数乘法+加法）中，AM↔PM 耦合**只来自加法步骤的矢量干涉**，而非乘法。双线性结构 \((W_1 z) \odot (W_2 z)\) 的乘法步骤本身也不增加任何新的 AM↔PM 耦合。

> **⭐ 关键观点 3：AM→PM 必须来自非线性——这是数学定理。**

对任何复数线性映射 \(f(z) = Wz\)，做纯模长缩放 \(z \to \alpha z\)（\(\alpha \in \mathbb{R}^+\)）：

\[
f(\alpha z) = \alpha \cdot Wz = \alpha \cdot f(z)
\]

输出只被实数缩放，相位完全不变。在任何线性系统中，**改变输入的"强度"不可能改变输出的"意义"**。AM→PM 耦合是本质上非线性的现象。

这意味着 PhaseTwist 要解决的问题是**真实存在**的——FFN 确实需要某种非线性来提供 AM↔PM 耦合。问题不在目标，而在实现方式。

---

## 四、对"解耦再操作"路线的否定

一度考虑的设计方案是：将复数显式分解为模长和相位，分别用实数网络处理后重组：

\[
\Delta\theta = W_{r\to\theta}\, r + b_\theta, \quad s = \sigma(W_{\theta\to r}\,[\cos\theta;\, \sin\theta] + b_r)
\]

> **⭐ 关键观点 4：解耦 \((r, \theta)\) 再用实数矩阵处理，等价于套了极坐标壳子的二维实数网络。**

复数的代数结构——乘法的旋转性、加法的干涉性——在拆解的那一刻全部丢失。这条路线放弃了复数域的核心优势，与项目"一切源自复数的代数结构"的哲学根本矛盾。

正确的方向是：寻找一种**不做任何解耦**的天然复数运算来实现 AM↔PM 耦合。

---

## 五、Cauchy-Riemann 方程的发现

> **⭐⭐⭐ 核心发现：Cauchy-Riemann 方程保证任何非幂函数的全纯映射天然具有 AM↔PM 耦合。**

对于任意全纯函数 \(f(z)\)，极坐标下的 Cauchy-Riemann 方程给出：

\[
\frac{\partial \Phi}{\partial r} = -\frac{1}{r}\frac{\partial \ln R}{\partial \theta}
\]

左边 \(\partial\Phi/\partial r\) 正是 **AM→PM 耦合强度**——输入模长变化引起输出相位变化。
右边 \(\partial\ln R/\partial\theta\) 正是 **PM→AM 耦合强度**——输入相位变化引起输出模长变化。

**两个耦合方向被调和共轭关系精确绑定。** 有一个就必然有另一个，且强度由复解析结构内禀决定。唯一不具有 AM↔PM 耦合的全纯函数是幂函数 \(f(z) = c \cdot z^n\)。

**结论：AM↔PM 不需要被"设计"进去，它是全纯非线性的固有性质。只需要施加任何一个非幂函数的全纯非线性，AM↔PM 就自动涌现，且两个方向的耦合形式由 Cauchy-Riemann 方程决定，不需要人为指定。**

---

## 六、Liouville 定理的约束与 RMSNorm 的解法

全纯非线性的根本困境：**Liouville 定理**——任何有界的全纯函数必为常数。不存在一个既全纯、又全局有界、又非常数的激活函数。这是复数域激活函数的根本难题，大多数复数网络论文因此放弃全纯性，退回到"拆成实部虚部分别做 ReLU"。

> **⭐ 关键观点 5：RMSNorm 将输入约束在有界区域内，使 Liouville 定理的限制可以被绕过。**

Anla 架构中 ComplexRMSNorm 保证了每一层输出的模长分布被归一化到 \(|z| \sim O(1)\) 附近。在这个有界圆盘内，低阶全纯多项式是安全且数值稳定的。问题从"找全局安全的全纯非线性"变为"在 RMSNorm 约束的有界区域内找表达力足够的全纯非线性"。

---

## 七、最终设计：\(f(z) = z + \alpha z^2\)

> **⭐⭐ 核心设计：HolomorphicActivation**

最低阶的非幂函数全纯非线性。展开其 AM-PM 性质（\(|\alpha r| \ll 1\) 近似）：

\[
|f| \approx r\big(1 + |\alpha| r \cos(\theta + \angle\alpha)\big), \quad \arg(f) \approx \theta + |\alpha| r \sin(\theta + \angle\alpha)
\]

**AM→PM**：\(\Delta\theta \approx |\alpha| r \sin(\theta + \angle\alpha)\)——相位偏移**同时取决于模长 \(r\) 和相位 \(\theta\)**。对比 PhaseTwist 的 \(\Delta\theta = \gamma r\)（只依赖模长，与相位无关）。"大声喊"的语义偏移取决于"说了什么"。

**PM→AM**：\(\Delta R/R \approx |\alpha| r \cos(\theta + \angle\alpha)\)——模长调制取决于相位。和 AM→PM 相差 \(\pi/2\)（\(\sin\) vs \(\cos\)），正是调和共轭对。不是人为设计，是解析结构的产物。

**跨维度 AM↔PM 耦合**不在激活函数层面实现，而是通过 FFN 的三段式结构自然涌现：W1 将 D 维信息线性混合到 D_ffn 维 → 逐元素全纯非线性在混合后的信号上施加 AM↔PM 耦合 → W2 将耦合后的信号投影回 D 维。跨维度耦合的路由完全由可学习权重 W1, W2 控制。

**频率创造**：\(z^2 = r^2 e^{2i\theta}\) 产生二次谐波。D=64 个基频经二次谐波扩展，理论上可产生 2080 个频率分量，远超 V=256 所需的 128 个。

---

## 八、PhaseTwist vs HolomorphicActivation 对比总览

| 维度 | PhaseTwist | HolomorphicActivation |
|------|-----------|----------------------|
| AM→PM 形式 | \(\gamma r\)（只依赖模长） | \(|\alpha|r\sin(\theta+\angle\alpha)\)（依赖模长**和**相位） |
| PM→AM 形式 | \(\beta\cos(\theta-\varphi)\)（固定余弦） | \(|\alpha|r\cos(\theta+\angle\alpha)\)（Cauchy-Riemann 绑定） |
| 两方向耦合关系 | 独立参数, 无数学约束 | 调和共轭对, 由 C-R 方程绑定 |
| 参数 | 3 个实数/通道 (\(\gamma,\beta,\varphi\)) | 1 个复数/通道 (\(\alpha\), 2 个实数自由度) |
| 跨维度交互 | 无（逐元素独立） | 通过 W1/W2 路由（数据驱动） |
| 旋转对称性 | 被 \(\varphi\) 打破 | 保持（全纯函数的等变性） |
| 频率创造 | 不产生新频率 | 产生 \(2\theta\) 二次谐波 |
| 反馈振荡风险 | 高（固定耦合 × 逐层叠加） | 低（\(\alpha\) 可学习, 可被梯度压至 0） |
| 全纯性 | 非全纯 (\(df/dz^* \neq 0\)) | 全纯 (\(df/dz^* = 0\)) |
| 反向传播复杂度 | 4 个偏导项 + 极坐标变换 | 1 个偏导项, 无需分解 |

---

## 九、代码变更

实际改动极小：

**`activation.py`**：新增 `HolomorphicActivation` 类。前向 `z * (1 + alpha * z)`，反向 `grad * conj(1 + 2*alpha*z)`。PhaseTwist 保留供向后兼容。

**`transformer_block.py`**：FFN 中 `PhaseTwist(self.ff_dim)` → `HolomorphicActivation(self.ff_dim)`。其余结构不变。

**`capacity_pressure_test_v5.py`**：无需修改。脚本与激活函数零耦合。

Wirtinger 梯度验证（float64）：输入梯度和参数梯度相对误差均 ~1e-7，PASS。

---

## 十、研究路径的关键转折点总结

1. **振荡归因**：从"可能是学习率/优化器问题"→ 定位到"PhaseTwist 的固定耦合是前向传播内的扰动放大器"。
2. **复数乘法的纠错**：从"复数乘法天然 AM-PM 耦合"→ 修正为"乘法中模长和相位完全正交，AM→PM 必须来自非线性"。
3. **解耦路线的否定**：从"拆成 \((r,\theta)\) 用实数网络做跨通道调制"→ 意识到这丧失了复数代数结构，退化为二维实数网络。
4. **Cauchy-Riemann 的核心发现**：AM↔PM 耦合是全纯函数的固有性质，不需要设计，只需要选择一个非幂函数的全纯非线性。
5. **Liouville 定理的绕过**：RMSNorm 提供的有界约束使得低阶全纯多项式成为安全的选择。
6. **最终收敛**：\(f(z) = z + \alpha z^2\)——最简洁的全纯二次非线性，一个公式同时解决了 AM↔PM 耦合、频率创造、训练稳定性三个问题。

