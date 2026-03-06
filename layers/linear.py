"""
保存位置: Anla/layers/linear.py

ComplexLinear — 复数线性层
=========================================================================

v4 变更:
    优化器: 内联 RMSProp → Polar Adam (极坐标 Adam)

    原版问题:
        · 用 |g|² 做自适应分母, 径向和切向梯度共享归一化
          → 如果相位梯度持续较大, 会压制模长学习率
        · 无一阶矩 (动量), 梯度不做平滑
        · 无偏差校正, init_energy=1e-3 是 ad-hoc 冷启动保护

    修正:
        · 极坐标分解: 将梯度投影到 (径向, 切向),
          分别维护独立的 Adam 一阶矩 + 二阶矩
        · 偏差校正替代 init_energy 冷启动保护
        · 动量平滑梯度方向, 尤其有益于相位梯度的稳定性
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, PolarAdamState
from Anla.utils.complex_ops import complex_kaiming_normal_


class ComplexLinear(ComplexLayer):
    """
    复数全连接层, 内置 Polar Adam 优化器。

    v4: 优化器从 RMSProp 升级为 Polar Adam,
        径向 (模长) 和切向 (相位) 梯度拥有独立的自适应学习率。
    """
    def __init__(self, in_features, out_features, bias=True, mode='descent'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        # 权重初始化
        weight_real = torch.empty(out_features, in_features)
        weight_imag = torch.empty(out_features, in_features)
        complex_kaiming_normal_(weight_real, weight_imag)
        self.weight = nn.Parameter(torch.complex(weight_real, weight_imag))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)

        # [v4] Polar Adam 优化器状态
        self._w_optim = PolarAdamState(shape=(out_features, in_features))
        if bias:
            self._b_optim = PolarAdamState(shape=(out_features,))
        else:
            self._b_optim = None

    def _ensure_optim_device(self):
        """确保优化器状态与参数在同一设备上。"""
        device = self.weight.device
        self._w_optim.to(device)
        if self._b_optim is not None:
            self._b_optim.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [v2] 安全缓存: detach().clone() 防止上游原地修改污染
        self.input_cache = x.detach().clone() if self.training else None
        with torch.no_grad():
            return nn.functional.linear(x, self.weight, self.bias)

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float,
                        weight_decay: float = 0.0) -> torch.Tensor:
        """
        手动反向传播: 计算梯度 → Polar Adam 更新权重。

        梯度计算 (Wirtinger):
            grad_input = grad_output @ W^*     (dL/dx* = dL/dy* · W^*)
            d_weight   = grad_output^T @ x^*   (dL/dW* = y^T · x^*)

        Polar Adam 更新:
            将 d_weight 投影到 W 的极坐标 (径向, 切向),
            独立 Adam 归一化后合成复数更新。

        模式说明:
            descent: W -= lr · polar_adam_step(dL/dW*)   (标准梯度下降)
            hebbian: W += lr · polar_adam_step(dL/dW*)   (Hebbian 学习)

            PolarAdamState.step 内部总是执行 param -= lr · step,
            所以 hebbian 模式传入取反的梯度以实现加法效果。
        """
        self._ensure_optim_device()
        x = self.input_cache

        # --- 梯度传播: 传给前层的梯度 ---
        grad_input = grad_output @ self.weight.conj()

        # --- 参数梯度计算 ---
        if x.dim() > 2:
            x_flat = x.reshape(-1, x.shape[-1])
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            x_flat = x
            grad_flat = grad_output

        scale = 1.0 / x_flat.shape[0]    # Batch 平均

        d_weight = (grad_flat.mT @ x_flat.conj()) * scale

        # --- Polar Adam 更新权重 ---
        # descent: 传原始梯度 → step 内部 sub_ → 实现 W -= step (最小化 loss)
        # hebbian: 传取反梯度 → step 内部 sub_ → 实现 W += step (Hebbian 学习)
        grad_for_optim = d_weight if self.mode == 'descent' else -d_weight
        self._w_optim.step(self.weight, grad_for_optim,
                           learning_rate, weight_decay)

        # --- Polar Adam 更新偏置 ---
        if self.bias is not None:
            d_bias = grad_flat.sum(dim=0) * scale
            bias_grad = d_bias if self.mode == 'descent' else -d_bias
            self._b_optim.step(self.bias, bias_grad,
                               learning_rate, 0.0)

        return grad_input
