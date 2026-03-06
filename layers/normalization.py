"""
保存位置: Anla/layers/normalization.py

复数 RMS Normalization
=========================================================================

v4 变更:
    scale 参数的优化器: RMSProp → 完整 Adam (一阶矩 + 偏差校正)
    通过 AdaptiveParamState 的内部升级自动获得, 本文件代码不变。
    · [+] 动量: 平滑 scale 的梯度, 减少训练初期的震荡
    · [+] 偏差校正: 替代 init_energy 冷启动保护
    · [Δ] β₂: 0.90 → 0.99

v2 变更:
    scale 参数的裸 SGD → AdaptiveParamState (当时为 RMSProp)

    前向和输入梯度数学完全不变。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, AdaptiveParamState


class ComplexRMSNorm(ComplexLayer):
    """
    复数 RMS Normalization

    v4: scale 参数的 AdaptiveParamState 内部升级为 Adam。
    v2: scale 参数配备 AdaptiveParamState, 替代裸 SGD。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.scale = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32))

        # [v4] Adam 优化器 (参数含动量 + 偏差校正)
        self._scale_optim = AdaptiveParamState(shape=(normalized_shape,))

    def _ensure_optim_device(self):
        self._scale_optim.to(self.scale.device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.input_cache = z.detach().clone()

        norm_z = torch.abs(z)
        rms = torch.sqrt(torch.mean(norm_z ** 2, dim=-1, keepdim=True) + self.eps)
        z_normalized = z / rms
        output = z_normalized * self.scale

        self.output_cache = (z_normalized, rms)
        return output

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        self._ensure_optim_device()

        z = self.input_cache
        z_normalized, rms = self.output_cache

        # scale 梯度
        grad_scale_per_element = 2.0 * torch.real(grad_output * torch.conj(z_normalized))
        grad_scale_flat = grad_scale_per_element.view(-1, self.normalized_shape)
        total_samples = grad_scale_flat.shape[0]
        grad_scale = torch.sum(grad_scale_flat, dim=0) / total_samples

        # [v4] Adam 更新 (含动量 + 偏差校正)
        self._scale_optim.step(self.scale, grad_scale, learning_rate)

        # 输入梯度 (完全不变)
        grad_y = grad_output * self.scale
        dot_prod = torch.real(
            torch.sum(grad_y * torch.conj(z_normalized), dim=-1, keepdim=True)
        )
        dim = self.normalized_shape
        numerator = grad_y - z_normalized * (dot_prod / dim)
        grad_input = numerator / rms

        self.clear_cache()
        return grad_input
