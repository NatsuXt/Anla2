"""
保存位置: Anla/layers/activation.py

全纯非线性激活函数 (Holomorphic Nonlinear Activation)
=========================================================================

v5 → v6 变更:
    [1] 新增 HolomorphicActivation, 取代 PhaseTwist 作为 FFN 的默认非线性。

        PhaseTwist 的问题 (v5 复盘):
            PhaseTwist 用固定的函数形式 (γ·r, β·cos(θ-φ)) 硬编码 AM↔PM 耦合。
            · γ, β 是全局标量, 所有 token、所有维度共享同一耦合强度。
            · 逐元素独立, 无跨维度交互。
            · 固定耦合在每次前向传播中无条件施加, 3 层 block 叠加后
              形成耦合振荡器反馈环路, 导致后期训练的系统性振荡。
            · φ 参数在复平面上定义特权方向, 打破旋转对称性。

        设计原理 — Cauchy-Riemann 方程的启示:
            对于任意全纯函数 f(z), 极坐标下 Cauchy-Riemann 方程给出:

                ∂Φ/∂r = -(1/r) · ∂(ln R)/∂θ

            左边 = AM→PM 耦合强度 (输入模长变化 → 输出相位变化)
            右边 = PM→AM 耦合强度 (输入相位变化 → 输出模长变化)

            两个耦合方向被调和共轭关系精确绑定:
            只要 f 不是幂函数 c·z^n, AM→PM 和 PM→AM 就同时存在,
            且强度由复解析结构内禀决定, 不需要人为设计。

        HolomorphicActivation: f(z) = z + α · z²

            最低阶的非幂函数全纯非线性。在 |z| ~ 1 的 RMSNorm 约束下:
            · AM→PM: arg(f) ≈ θ + |α|·r·sin(θ + ∠α)
                     模长 r 的变化引起相位偏移, 且偏移方向取决于当前相位 θ。
                     ("大声喊"的语义偏移取决于"说了什么")
            · PM→AM: |f| ≈ r·(1 + |α|·r·cos(θ + ∠α))
                     相位 θ 的值调制输出模长增益。
                     ("不同语义天然具有不同的表达强度")
            · 两个方向的耦合由同一个复数参数 α 的模长和相位控制,
              满足 Cauchy-Riemann 约束。

            对比 PhaseTwist:
                PhaseTwist AM→PM: Δθ = γ·r      (只依赖模长, 与相位无关)
                Holomorphic AM→PM: Δθ ≈ |α|·r·sin(θ+∠α)  (同时依赖模长和相位)

                PhaseTwist PM→AM: Δr/r = β·cos(θ-φ)  (固定余弦, 一个自由度)
                Holomorphic PM→AM: ΔR/R ≈ |α|·r·cos(θ+∠α)  (由 α 的两个自由度控制)

            跨维度耦合:
                HolomorphicActivation 本身是逐元素的。跨维度 AM→PM 耦合
                通过 FFN 的 W1 → HolomorphicAct → W2 结构实现:
                    W1: 将 D 维输入的模长/相位信息混合到 D_ffn 维隐层
                    Act: 在每个混合后的隐层分量上施加全纯 AM↔PM 耦合
                    W2: 将耦合后的信号投影回 D 维
                维度 j 的模长 → (W1 混合) → 隐层 m 的模长 → (Act 耦合) →
                隐层 m 的相位 → (W2 混合) → 维度 k 的相位。
                跨维度耦合的路由由 W1, W2 的可学习权重控制。

            频率创造:
                z² = r²·e^{2iθ} 产生 2θ 的二次谐波。
                D=64 个基频经过二次谐波扩展, 理论上可产生最多
                64 + C(64,2) = 2080 个频率分量, 远超 V=256 的环
                所需的 128 个频率。3 层 block 的级联进一步将
                频率生成能力提升到 2^3=8 次谐波。

            数值安全:
                Liouville 定理: 不存在全局有界的非常数全纯函数。
                但 RMSNorm 将 W1 的输出约束在 |z| ~ O(1) 附近,
                在此有界区域内 z² ~ O(1), α·z² ~ O(0.1),
                整个函数在 RMSNorm 约束下数值稳定。

            反向传播 (Wirtinger):
                f 全纯 ⟹ df/dz* = 0, 链式法则简化为:
                    dL/dz* = (dL/df*) · conj(df/dz)
                             = grad_output · conj(1 + 2α·z)
                参数梯度:
                    dL/dα* = Σ (dL/df*) · conj(z²)
                             = Σ grad_output · conj(z²)
                比 PhaseTwist 的反向传播简洁得多 (无需极坐标分解)。

    [保留] PhaseTwist 类不删除, 供旧实验和测试的向后兼容。
"""

import math
import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, AdaptiveParamState, PolarAdamState


# ========== 统一的数值常量 ==========
EPS = 1e-7
EPS_SAFE = 1e-4


# =====================================================================
#  v6 默认激活: HolomorphicActivation
# =====================================================================
class HolomorphicActivation(ComplexLayer):
    """
    全纯非线性激活函数。

    f(z) = z + α · z²

    其中 α ∈ C^{channels} 是逐通道可学习的复数参数。

    数学性质:
        · 全纯: df/dz* = 0, Cauchy-Riemann 方程成立
        · AM→PM: arg(f) 依赖于 |z| (通过 Cauchy-Riemann 自然涌现)
        · PM→AM: |f| 依赖于 arg(z) (通过 Cauchy-Riemann 自然涌现)
        · 两个方向的耦合强度由调和共轭关系精确绑定
        · α 的模长控制耦合强度, α 的相位控制耦合方向
        · 逐通道独立的 α_d 提供 per-dim 差异化的耦合特性

    参数:
        channels:  通道数 (= FFN 隐层维度 D_ffn)
        init_mag:  α 的初始模长, 控制初始非线性强度。
                   默认 0.1, 在 RMSNorm 约束下 (|z| ~ 1) 对应:
                   · 相位偏移 ~ 0.1 rad ≈ 5.7° (足以提供初始非线性)
                   · 模长调制 ~ ±10% (温和的初始门控)
                   · 远小于不稳定临界点 |α|·|z| = 0.5
    """

    def __init__(self, channels: int, init_mag: float = 0.1):
        super().__init__()
        self.channels = channels

        # ---------------------------------------------------------------
        # α ∈ C^{channels}: 逐通道复数耦合系数
        #
        # 初始化: 模长 = init_mag, 相位在 [0, 2π) 上均匀随机。
        # 随机相位打破通道间对称性, 使不同维度从训练开始就探索
        # 不同方向的 AM↔PM 耦合, 避免对称性冗余。
        # ---------------------------------------------------------------
        phase = torch.rand(channels) * 2.0 * math.pi
        alpha_real = init_mag * torch.cos(phase)
        alpha_imag = init_mag * torch.sin(phase)
        self.alpha = nn.Parameter(
            torch.complex(alpha_real, alpha_imag)
        )

        # Polar Adam 优化器: α 是复数参数, 径向 (|α|) 和切向 (∠α)
        # 应有独立的自适应学习率。径向控制耦合强度, 切向控制耦合方向。
        self._alpha_optim = PolarAdamState(shape=(channels,))

    def _ensure_optim_device(self):
        """确保优化器状态与参数在同一设备上。"""
        self._alpha_optim.to(self.alpha.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向: 全纯二次非线性。

        f(z) = z + α · z²
             = z · (1 + α · z)

        等价形式 z·(1+α·z) 揭示其本质: 对输入 z 施加一个
        数据依赖的复数门控 (1+α·z), 该门控的模长和相位
        同时取决于 z 的模长和相位 (AM↔PM 耦合)。

        输入:
            z: [..., channels] 复数张量 (来自 ff1 线性层的输出)
        输出:
            f(z): [..., channels] 复数张量, 形状不变
        """
        # 缓存输入供反向传播使用
        self.input_cache = z.detach().clone() if self.training else None

        # alpha 形状 [channels], 自动广播到 z 的 [..., channels]
        alpha = self.alpha

        # f(z) = z · (1 + α · z)
        # 两次复数乘法, 一次复数加法, 无极坐标分解
        output = z * (1.0 + alpha * z)

        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        手动反向传播。

        因为 f 是全纯的, 反向传播极为简洁:

        ┌─────────────────────────────────────────────────────┐
        │  f 全纯 ⟹ df/dz* = 0                              │
        │                                                     │
        │  输入梯度:                                           │
        │    dL/dz* = grad_output · conj(df/dz)               │
        │           = grad_output · conj(1 + 2α·z)            │
        │                                                     │
        │  参数梯度:                                           │
        │    dL/dα* = Σ_{batch} grad_output · conj(z²)        │
        └─────────────────────────────────────────────────────┘

        推导:
            Wirtinger 链式法则 (L 为实值损失):
                dL/dz* = (dL/df*) · conj(df/dz) + (dL/df) · (df/dz*)
                                                       ^^^^^^^^^^^
                                                       = 0 (f 全纯)
                       = grad_output · conj(1 + 2α·z)

            参数梯度 (f 关于 α 也是全纯的, df/dα* = 0):
                dL/dα* = (dL/df*) · (df*/dα*)
                       = grad_output · conj(df/dα)
                       = grad_output · conj(z²)

            对比 PhaseTwist 的反向传播:
                PhaseTwist 非全纯 ⟹ df/dz* ≠ 0, 需要同时计算
                df/dz 和 df/dz*, 再做极坐标到 Wirtinger 的变换,
                产生 4 个偏导项的组合。
                HolomorphicActivation 全纯 ⟹ df/dz* = 0,
                只需 1 个偏导项, 计算量和代码复杂度大幅降低。
        """
        z = self.input_cache
        if z is None:
            return grad_output

        self._ensure_optim_device()

        alpha = self.alpha

        # ---------------------------------------------------------------
        # Step 1: 输入梯度
        #   df/dz = 1 + 2α·z  (全纯导数, 唯一需要的 Wirtinger 分量)
        #   dL/dz* = grad_output · conj(1 + 2α·z)
        # ---------------------------------------------------------------
        df_dz = 1.0 + 2.0 * alpha * z
        grad_input = grad_output * df_dz.conj()

        # ---------------------------------------------------------------
        # Step 2: 参数梯度
        #   df/dα = z²  (逐通道, f 关于 α 全纯)
        #   dL/dα* = Σ_{batch,seq} grad_output · conj(z²)
        #
        #   对 batch 和序列维度求和, 保留通道维度。
        #   除以样本数做 batch 平均, 与 ComplexLinear 的约定一致。
        # ---------------------------------------------------------------
        z_sq = z * z
        d_alpha_elem = grad_output * z_sq.conj()

        total_samples = z.numel() // z.shape[-1]
        d_alpha = d_alpha_elem.reshape(-1, self.channels).sum(dim=0) / total_samples

        # ---------------------------------------------------------------
        # Step 3: Polar Adam 更新 α
        #   α 是复数参数, PolarAdamState 将梯度 dL/dα* 分解为:
        #     径向分量 (控制 |α|, 即耦合强度)
        #     切向分量 (控制 ∠α, 即耦合方向)
        #   分别维护独立的 Adam 矩, 避免强度和方向的优化互相干扰。
        # ---------------------------------------------------------------
        self._alpha_optim.step(self.alpha, d_alpha, learning_rate)

        self.clear_cache()
        return grad_input


# =====================================================================
#  v5 遗留: PhaseTwist (向后兼容, 不再作为 FFN 默认激活)
# =====================================================================
class PhaseTwist(ComplexLayer):
    """
    [v5 遗留] 双向耦合复数激活函数 (无 tanh 压缩)。

    v6 说明:
        此类保留供旧实验脚本和测试的向后兼容。
        v6 起, FFN 的默认激活函数为 HolomorphicActivation。
        PhaseTwist 的设计问题详见文件头部文档。

    v5: γ/β 初始值提升 10 倍, 激活初始非线性。
    v4: 移除 tanh 模长压缩, 模长自由通过, 仅由 PM→AM 耦合调制。
        非线性完全由 AM→PM 相位旋转 (θ + γ·r) 提供。
    """

    def __init__(self, channels: int,
                 init_gamma: float = 0.1,
                 init_beta: float = 0.1,
                 init_phi: float = 0.0):
        super().__init__()
        self.channels = channels

        self.gamma = nn.Parameter(
            torch.full((channels,), init_gamma, dtype=torch.float32))
        self.beta = nn.Parameter(
            torch.full((channels,), init_beta, dtype=torch.float32))
        self.phi = nn.Parameter(
            torch.full((channels,), init_phi, dtype=torch.float32))

        # Adam 优化器
        self._gamma_optim = AdaptiveParamState(shape=(channels,))
        self._beta_optim = AdaptiveParamState(shape=(channels,))
        self._phi_optim = AdaptiveParamState(shape=(channels,))

    def _ensure_optim_device(self):
        device = self.gamma.device
        self._gamma_optim.to(device)
        self._beta_optim.to(device)
        self._phi_optim.to(device)

    def _broadcast(self, param: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = [1] * (ndim - 1) + [self.channels]
        return param.view(*shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向: 幅相耦合非线性变换 (无 tanh 压缩)

        f(z) = m · e^{i·(θ + γ·r)}

        其中 m = r · (1 + β · cos(θ - φ)):
            · r > 0: 输入模长
            · β · cos(θ - φ): 相位相关的增益/衰减 (PM → AM)
            · γ · r: 模长相关的相位旋转 (AM → PM)
        """
        self.input_cache = z.detach().clone() if self.training else None

        r = torch.abs(z) + EPS
        theta = torch.angle(z)

        gamma = self._broadcast(self.gamma, z.dim())
        beta = self._broadcast(self.beta, z.dim())
        phi = self._broadcast(self.phi, z.dim())

        # PM → AM 耦合: 相位调制模长
        cos_diff = torch.cos(theta - phi)
        m = r * (1.0 + beta * cos_diff)

        # 防止 m < 0 (当 β > 1 且 cos_diff ≈ -1 时可能触发)
        m = torch.clamp(m, min=EPS)

        # AM → PM 耦合: 模长驱动相位旋转
        theta_out = theta + gamma * r

        # 直接合成, 不经过 tanh 压缩
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        output = m * e_i_tout

        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        手动反向传播 (无 tanh 版本)。

        前向: f = m · e^{iθ_out}
              m = r·(1+β·cos(θ-φ)),  θ_out = θ+γ·r

        极坐标偏导:
            df/dr    = (1+β·cos(θ-φ)) · e^{iθ_out} + i·γ·f
            df/dθ    = -r·β·sin(θ-φ) · e^{iθ_out} + i·f

        参数梯度:
            df/dγ    = i·f·r
            df/dβ    = r·cos(θ-φ) · e^{iθ_out}
            df/dφ    = r·β·sin(θ-φ) · e^{iθ_out}
        """
        z = self.input_cache
        if z is None:
            return grad_output

        self._ensure_optim_device()

        r = torch.abs(z) + EPS
        theta = torch.angle(z)

        gamma = self._broadcast(self.gamma, z.dim())
        beta = self._broadcast(self.beta, z.dim())
        phi = self._broadcast(self.phi, z.dim())

        cos_diff = torch.cos(theta - phi)
        sin_diff = torch.sin(theta - phi)
        m = r * (1.0 + beta * cos_diff)
        m = torch.clamp(m, min=EPS)
        theta_out = theta + gamma * r
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        f = m * e_i_tout

        # Step 1: 极坐标偏导数
        df_dr = (1.0 + beta * cos_diff) * e_i_tout + 1j * gamma * f
        df_dtheta = -r * beta * sin_diff * e_i_tout + 1j * f

        # Step 2: Wirtinger 变换
        z_hat = z / r
        z_hat_c = z_hat.conj()
        safe_inv_r = 1.0 / torch.clamp(r, min=EPS_SAFE)

        df_dz = df_dr * (0.5 * z_hat_c) + df_dtheta * (-0.5j * safe_inv_r * z_hat_c)
        df_dz_conj = df_dr * (0.5 * z_hat) + df_dtheta * (0.5j * safe_inv_r * z_hat)

        # Step 3: 输入梯度
        grad_input = (
            torch.conj(grad_output) * df_dz_conj
            + grad_output * torch.conj(df_dz)
        )

        # Step 4: 参数梯度
        df_dgamma = 1j * f * r
        d_gamma_elem = 2.0 * torch.real(grad_output * torch.conj(df_dgamma))

        df_dbeta = r * cos_diff * e_i_tout
        d_beta_elem = 2.0 * torch.real(grad_output * torch.conj(df_dbeta))

        df_dphi = r * beta * sin_diff * e_i_tout
        d_phi_elem = 2.0 * torch.real(grad_output * torch.conj(df_dphi))

        # Step 5: 聚合 & 更新
        total_samples = z.numel() // z.shape[-1]

        d_gamma = d_gamma_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_beta = d_beta_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_phi = d_phi_elem.reshape(-1, self.channels).sum(dim=0) / total_samples

        # Adam 更新
        self._gamma_optim.step(self.gamma, d_gamma, learning_rate)
        self._beta_optim.step(self.beta, d_beta, learning_rate)
        self._phi_optim.step(self.phi, d_phi, learning_rate)

        self.clear_cache()
        return grad_input
