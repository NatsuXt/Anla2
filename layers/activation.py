"""
保存位置: Anla/layers/activation.py

双向耦合激活函数 (Bidirectional AM-PM / PM-AM Coupling Activation)
=========================================================================

v4 → v5 变更:
    [1] PhaseTwist 默认初始化: init_gamma=0.01 → 0.1, init_beta=0.01 → 0.1

        动机 (来自理论分析):
            v4 的 γ=0.01 在 RMSNorm 后模长约 1.0 的条件下,
            PhaseTwist 的相位旋转量 γ·r ≈ 0.01 rad ≈ 0.57°。
            这使得 PhaseTwist 在训练初期几乎是线性的,
            非线性效应需要等 γ 被梯度逐步增大后才能生效。

            实数网络中 GELU/ReLU 从第一步就有约 50% 的输入
            被"激活"或"截断", 非线性从训练一开始就发挥作用。

            v5 将 γ 提升到 0.1, 对应 ≈ 5.7° 的初始旋转:
                · 提供足够的初始非线性 (cos(θ+0.1r) vs cos(θ) 有可测差异)
                · 梯度中 |iγf| ≈ 0.1|f|, 约占直通项的 10%, 不会主导
                · 处于"可恢复区间" — 若过大, Adam 有能力调回

            β 同步提升到 0.1:
                · PM→AM 耦合从 1±0.01 (几乎无效) 到 1±0.1 (10% 方向选择性)
                · 远低于临界值 β=1 (此时 m=r(1+β·cos)=0 可能触发)

    其余与 v4 完全一致。
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

    v5: γ/β 初始值提升 10 倍, 激活初始非线性。
    v4: 移除 tanh 模长压缩, 模长自由通过, 仅由 PM→AM 耦合调制。
        非线性完全由 AM→PM 相位旋转 (θ + γ·r) 提供。
    """

    def __init__(self, channels: int,
                 init_gamma: float = 0.1,     # [v5] 0.01 → 0.1
                 init_beta: float = 0.1,      # [v5] 0.01 → 0.1
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

        # [v4.4] 防止 m < 0 (当 β > 1 且 cos_diff ≈ -1 时可能触发)
        m = torch.clamp(m, min=EPS)

        # AM → PM 耦合: 模长驱动相位旋转
        theta_out = theta + gamma * r

        # [v4] 直接合成, 不经过 tanh 压缩
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        output = m * e_i_tout

        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        手动反向传播 (无 tanh 版本)。与 v4 完全一致。

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
        m = torch.clamp(m, min=EPS)                    # [v4.4] 与前向一致
        theta_out = theta + gamma * r
        e_i_tout = torch.polar(torch.ones_like(theta_out), theta_out)
        f = m * e_i_tout                           # [v4] m 替代 tanh(m)

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

        # [v4] Adam 更新
        self._gamma_optim.step(self.gamma, d_gamma, learning_rate)
        self._beta_optim.step(self.beta, d_beta, learning_rate)
        self._phi_optim.step(self.phi, d_phi, learning_rate)

        self.clear_cache()
        return grad_input
