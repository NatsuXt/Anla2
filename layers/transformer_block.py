"""
保存位置: Anla/layers/transformer_block.py

复数 Transformer Block
=========================================================================

v5 → v6 变更:
    [1] FFN 激活函数: PhaseTwist → HolomorphicActivation

        旧 (v5): FFN = ComplexLinear → PhaseTwist → ComplexLinear
        新 (v6): FFN = ComplexLinear → HolomorphicActivation → ComplexLinear

        PhaseTwist 用固定函数形式 (γ·r, β·cos(θ-φ)) 硬编码 AM↔PM 耦合,
        导致后期训练振荡 (固定耦合形成反馈环路)。

        HolomorphicActivation 使用全纯函数 f(z) = z + α·z²,
        AM↔PM 耦合由 Cauchy-Riemann 方程保证自然涌现:
            · AM→PM: arg(f) ≈ θ + |α|·r·sin(θ+∠α) (模长变化引起相位偏移)
            · PM→AM: |f| ≈ r·(1 + |α|·r·cos(θ+∠α)) (相位调制模长增益)
        耦合的强度和方向由逐通道可学习的复数参数 α 控制。

        跨维度 AM↔PM 耦合由 FFN 的三段式结构实现:
            W1: D → D_ffn (跨维度线性混合, 将所有维度的信息路由到隐层)
            Act: 逐元素全纯非线性 (在混合后的信号上施加 AM↔PM 耦合)
            W2: D_ffn → D (将耦合后的信号投影回原始维度)

        效果:
            维度 j 的模长 → W1 混合到隐层 m → Act 在 m 上做 AM→PM →
            隐层 m 的相位 → W2 投影到维度 k 的相位。
            跨维度耦合的路由完全由 W1, W2 的权重决定, 数据驱动。

    其余结构 (Pre-Norm, 残差连接, HolographicAttention) 不变。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.layers.linear import ComplexLinear
from Anla.layers.normalization import ComplexRMSNorm
from Anla.layers.activation import HolomorphicActivation
from Anla.layers.holographic_attention import HolographicAttention


class ComplexTransformerBlock(ComplexLayer):
    """
    [Anla 核心计算单元]

    结构: Pre-Norm Architecture
        x → Norm → HolographicAttention → + → Residual
        x → Norm → FFN (Linear → HolomorphicAct → Linear) → + → Residual

    v6: FFN 中的 PhaseTwist 替换为 HolomorphicActivation。
        · 全纯非线性: f(z) = z + α·z², AM↔PM 由 Cauchy-Riemann 保证
        · 跨维度耦合: W1 线性混合 → 逐元素全纯非线性 → W2 线性混合
        · 参数更少: 每通道 1 个复数 α (2 个实数自由度)
                    vs PhaseTwist 的 γ, β, φ (3 个实数自由度)
        · 反向传播更简洁: f 全纯 ⟹ df/dz* = 0, 省去半数导数项
    """
    def __init__(self, d_model, num_heads=4, ff_mult=4, dropout=0.0):
        super().__init__()

        # ---------------------------------------------------------------
        # Sub-layer 1: Attention
        # ---------------------------------------------------------------
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = HolographicAttention(d_model, num_heads=num_heads)

        # ---------------------------------------------------------------
        # Sub-layer 2: FFN
        #
        # v6 FFN 结构: ComplexLinear → HolomorphicActivation → ComplexLinear
        #
        # 跨维度 AM↔PM 耦合的信息流:
        #   ff1 (D→D_ffn):  将 D 个维度的模长+相位信息线性混合到 D_ffn 维
        #   act (D_ffn):     在每个混合后的隐层分量上施加全纯 AM↔PM 耦合
        #   ff2 (D_ffn→D):  将耦合后的 D_ffn 维信号投影回 D 维
        #
        # RMSNorm (norm2) 在 ff1 之前将输入归一化到 |z| ~ O(1),
        # 保证 ff1 输出在 Kaiming 初始化下 |h| ~ O(1),
        # 从而 HolomorphicActivation 的 z² 项数值安全。
        # ---------------------------------------------------------------
        self.norm2 = ComplexRMSNorm(d_model)
        self.ff_dim = d_model * ff_mult

        self.ff1 = ComplexLinear(d_model, self.ff_dim)
        self.act = HolomorphicActivation(self.ff_dim)
        self.ff2 = ComplexLinear(self.ff_dim, d_model)

        # Cache for residual connections
        self.res1_cache = None
        self.res2_cache = None

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        self.input_cache = x.detach().clone() if self.training else None

        # --- Block 1: Attention ---
        # Pre-Norm
        norm_x = self.norm1.forward(x)

        # Attention + Residual
        attn_out = self.attn.forward(norm_x, mask=mask)
        x = x + attn_out

        if self.training:
            self.res1_cache = x.detach().clone()  # Post-Attn Residual

        # --- Block 2: FFN ---
        # Pre-Norm
        norm_x2 = self.norm2.forward(x)

        # Feed Forward: Linear → HolomorphicAct → Linear
        ff_h = self.ff1.forward(norm_x2)
        ff_h = self.act.forward(ff_h)
        ff_out = self.ff2.forward(ff_h)

        # Residual
        x = x + ff_out

        return x

    def manual_backward(self, grad_output: torch.Tensor, lr: float, wd: float = 0.0) -> torch.Tensor:
        """
        手动反向传播流：逆序穿过 FFN → Norm2 → Attn → Norm1

        梯度流 (v6 与 v5 结构完全一致, 仅 act 层的内部计算不同):
            grad_output
                │
                ├──→ ff2.backward → act.backward → ff1.backward → norm2.backward
                │                                                        │
                ├────────────────────────────────────── + ←───────────────┘
                │ (Residual 1 merge)
                │
                ├──→ attn.backward → norm1.backward
                │                          │
                ├──────────────── + ←───────┘
                │ (Input Residual merge)
                ▼
            grad_input
        """
        # --- Backprop through Block 2 (FFN) ---
        # Residual split: 梯度同时流向 FFN 和 Skip Connection
        # FFN path
        grad_ff2 = self.ff2.manual_backward(grad_output, lr, wd)
        grad_act = self.act.manual_backward(grad_ff2, lr)  # HolomorphicAct 无 weight decay
        grad_ff1 = self.ff1.manual_backward(grad_act, lr, wd)

        # Norm2 path
        grad_norm2 = self.norm2.manual_backward(grad_ff1, lr)

        # Merge at Residual 1
        grad_res1 = grad_output + grad_norm2

        # --- Backprop through Block 1 (Attention) ---
        # Attention path
        grad_attn = self.attn.manual_backward(grad_res1, lr, wd)

        # Norm1 path
        grad_norm1 = self.norm1.manual_backward(grad_attn, lr)

        # Merge at Input Residual
        grad_input = grad_res1 + grad_norm1

        # Clear caches
        self.res1_cache = None
        self.input_cache = None

        return grad_input
