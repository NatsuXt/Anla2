import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer
from Anla.layers.linear import ComplexLinear
from Anla.layers.normalization import ComplexRMSNorm
from Anla.layers.activation import PhaseTwist
from Anla.layers.holographic_attention import HolographicAttention

class ComplexTransformerBlock(ComplexLayer):
    """
    [Anla 核心计算单元]
    结构: Pre-Norm Architecture
    x -> Norm -> HolographicAttention -> + -> Residual
    x -> Norm -> FFN (Linear->PhaseTwist->Linear) -> + -> Residual
    """
    def __init__(self, d_model, num_heads=4, ff_mult=4, dropout=0.0):
        super().__init__()
        
        # Sub-layer 1: Attention
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = HolographicAttention(d_model, num_heads=num_heads)
        
        # Sub-layer 2: FFN
        # FFN 在 Anla 中是“频率混合器”，PhaseTwist 提供非线性耦合
        self.norm2 = ComplexRMSNorm(d_model)
        self.ff_dim = d_model * ff_mult
        
        self.ff1 = ComplexLinear(d_model, self.ff_dim)
        self.act = PhaseTwist(self.ff_dim)
        self.ff2 = ComplexLinear(self.ff_dim, d_model)
        
        # Cache for residual connections
        self.res1_cache = None
        self.res2_cache = None # Usually same as input to ffn

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        self.input_cache = x.detach().clone() if self.training else None
        
        # --- Block 1: Attention ---
        # Pre-Norm
        norm_x = self.norm1.forward(x)
        
        # Attention + Residual
        attn_out = self.attn.forward(norm_x, mask=mask)
        x = x + attn_out
        
        if self.training:
            self.res1_cache = x.detach().clone() # Post-Attn Residual
            
        # --- Block 2: FFN ---
        # Pre-Norm
        norm_x2 = self.norm2.forward(x)
        
        # Feed Forward
        ff_h = self.ff1.forward(norm_x2)
        ff_h = self.act.forward(ff_h)
        ff_out = self.ff2.forward(ff_h)
        
        # Residual
        x = x + ff_out
        
        return x

    def manual_backward(self, grad_output: torch.Tensor, lr: float, wd: float = 0.0) -> torch.Tensor:
        """
        手动反向传播流：逆序穿过 FFN -> Norm2 -> Attn -> Norm1
        """
        # grad_output 是这一层输出的误差向量
        
        # --- Backprop through Block 2 (FFN) ---
        # Residual split: 梯度同时流向 FFN 和 Skip Connection
        # d(x + FFN)/dx = 1 + dFFN/dx
        # 此时 grad_output 直接流过 Skip connection 成为一部分 grad_x
        
        # FFN path
        grad_ff2 = self.ff2.manual_backward(grad_output, lr, wd)
        grad_act = self.act.manual_backward(grad_ff2, lr) # Act通常没有weight decay
        grad_ff1 = self.ff1.manual_backward(grad_act, lr, wd)
        
        # Norm2 path
        grad_norm2 = self.norm2.manual_backward(grad_ff1, lr)
        
        # Merge at Residual 1
        # 这里的梯度是: (来自下一层的梯度) + (来自 FFN Norm 的梯度)
        grad_res1 = grad_output + grad_norm2
        
        # --- Backprop through Block 1 (Attention) ---
        # Attention path
        grad_attn = self.attn.manual_backward(grad_res1, lr, wd)
        
        # Norm1 path
        grad_norm1 = self.norm1.manual_backward(grad_attn, lr)
        
        # Merge at Input Residual
        # grad_input = (grad_res1) + (grad_norm1)
        grad_input = grad_res1 + grad_norm1
        
        # Clear caches
        self.res1_cache = None
        self.input_cache = None
        
        return grad_input
