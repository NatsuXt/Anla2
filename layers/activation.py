"""
保存位置: Anla/layers/activation.py

双向耦合激活函数 (Bidirectional AM-PM / PM-AM Coupling Activation)
=========================================================================

v4 变更:
    [架构] 移除 tanh 模长压缩

    原版:
        m = r · (1 + β·cos(θ-φ))
        r_out = tanh(m)              ← 将模长压到 (0,1)
        θ_out = θ + γ·r

    问题:
        1. tanh(m) 将所有模长压到 <1, 下游 ff2 永远收到 "弱信号"
        2. 反向传播: sech²(m) 对大模长信号梯度接近 0 → 系统性梯度消失
        3. 动量累积此一致收缩信号 → EmbRMS 单调漂移 (根本原因)
        4. 实数网络中 GELU/SiLU 不压缩模长, 这才是健康的设计

    修正:
        m = r · (1 + β·cos(θ-φ))
        r_out = m                    ← 模长自由通过, 由 PM→AM 耦合调制
        θ_out = θ + γ·r             ← 不变

    非线性来源 (移除 tanh 后仍有充分的非线性):
        · AM→PM 耦合: θ_out = θ + γ·r
          展开: Re(output) = m·cos(θ+γr), Im(output) = m·sin(θ+γr)
          r 出现在 cos/sin 的参数里 → 对实部虚部都是强非线性
        · PM→AM 耦合: m = r·(1 + β·cos(θ-φ))
          特定相位增益/衰减 → 方向选择性

    防爆保障 (不需要 tanh):
        · L_Elegant 径向力: ln(r/r̂) → 拉向 target 模长
        · Weight decay: 权重收缩 → 间接约束信号幅度
        · RMSNorm: 每层归一化 → 防止逐层放大

    梯度改善:
        旧版 df/dr = sech²(m)·(1+β·cos)·e^{iθ_out} + ...
        新版 df/dr =         (1+β·cos)·e^{iθ_out} + ...
        移除 sech²(m) 后梯度不再被系统性压缩,
        从 FFN 回传到 Embedding 的梯度幅度由信号本身决定, 不受人为衰减。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, AdaptiveParamState


# ========== 统一的数值常量 ==========
EPS = 1e-7
EPS_SAFE = 1e-4


class PhaseTwist(ComplexLayer):
    """
    双向耦合复数激活函数 (无 tanh 压缩)。

    v4: 移除 tanh 模长压缩, 模长自由通过, 仅由 PM→AM 耦合调制。
        非线性完全由 AM→PM 相位旋转 (θ + γ·r) 提供。
    """

    def __init__(self, channels: int,
                 init_gamma: float = 0.01,
                 init_beta: float = 0.01,
                 init_phi: float = 0.0):
        super().__init__()
        self.channels = channels

        self.gamma = nn.Parameter(
            torch.full((channels,), init_gamma, dtype=torch.float32))
        self.beta = nn.Parameter(
            torch.full((channels,), init_beta, dtype=torch.float32))
        self.phi = nn.Parameter(
            torch.full((channels,), init_phi, dtype=torch.float32))

        # [v4] Adam 优化器
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

        # AM → PM 耦合: 模长驱动相位旋转
        theta_out = theta + gamma * r

        # [v4] 直接合成, 不经过 tanh 压缩
        # m 可以为负 (当 β > 1 且 cos_diff ≈ -1 时), 自然处理:
        # m * e^{iθ} = |m| * e^{i(θ+π)} 当 m < 0, 相当于模长取绝对值 + 相位翻转
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        output = m * e_i_tout

        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        手动反向传播 (无 tanh 版本)。

        前向: f = m · e^{iθ_out}
              m = r·(1+β·cos(θ-φ)),  θ_out = θ+γ·r

        极坐标偏导 (对比旧版, 所有 sech²(m) 因子移除):
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
        theta_out = theta + gamma * r
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        f = m * e_i_tout                           # [v4] m 替代 tanh(m)

        # Step 1: 极坐标偏导数 — [v4] 移除所有 sech²(m) 因子
        #
        # 旧版: df_dr = sech²·(1+β·cos)·e + iγf    (sech² 压缩梯度)
        # 新版: df_dr =       (1+β·cos)·e + iγf    (梯度自由流过)
        df_dr = (1.0 + beta * cos_diff) * e_i_tout + 1j * gamma * f
        df_dtheta = -r * beta * sin_diff * e_i_tout + 1j * f

        # Step 2: Wirtinger 变换 (与旧版完全一致)
        z_hat = z / r
        z_hat_c = z_hat.conj()
        safe_inv_r = 1.0 / torch.clamp(r, min=EPS_SAFE)

        df_dz = df_dr * (0.5 * z_hat_c) + df_dtheta * (-0.5j * safe_inv_r * z_hat_c)
        df_dz_conj = df_dr * (0.5 * z_hat) + df_dtheta * (0.5j * safe_inv_r * z_hat)

        # Step 3: 输入梯度 (与旧版完全一致)
        grad_input = (
            torch.conj(grad_output) * df_dz_conj
            + grad_output * torch.conj(df_dz)
        )

        # Step 4: 参数梯度 — [v4] 移除 sech²(m)
        df_dgamma = 1j * f * r
        d_gamma_elem = 2.0 * torch.real(grad_output * torch.conj(df_dgamma))

        df_dbeta = r * cos_diff * e_i_tout          # [v4] 无 sech²
        d_beta_elem = 2.0 * torch.real(grad_output * torch.conj(df_dbeta))

        df_dphi = r * beta * sin_diff * e_i_tout    # [v4] 无 sech²
        d_phi_elem = 2.0 * torch.real(grad_output * torch.conj(df_dphi))

        # Step 5: 聚合 & 更新 (与旧版完全一致)
        total_samples = z.numel() // z.shape[-1]

        d_gamma = d_gamma_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_beta = d_beta_elem.reshape(-1, self.channels).sum(dim=0) / total_samples
        d_phi = d_phi_elem.reshape(-1, self.channels).sum(dim=0) / total_samples

        # [v4] Adam 更新
        self._gamma_optim.step(self.gamma, d_gamma, learning_rate)
        self._beta_optim.step(self.beta, d_beta, learning_rate)
        self._phi_optim.step(self.phi, d_phi, learning_rate)

        self.clear_cache()
        return grad_input
