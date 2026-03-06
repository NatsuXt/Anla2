import torch
import torch.nn as nn
import math

class ComplexRotaryEmbedding(nn.Module):
    """
    [Anla Time-Space Clock] 复数旋转位置编码
    
    原理：
    与其做加法 (x + pos)，不如利用复数特性做乘法 (x * e^{i*theta})。
    这相当于给每个 Token 挂上了一个"时钟"，不同维度的时钟转速不同。
    
    物理含义：
    Input: z = r * e^{i*phi} (语义相位)
    Output: z' = r * e^{i*(phi + pos * freq)} (语义相位 + 时空相位)
    """
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        
        # 1. 计算频率 (Frequencies)
        # 按照 Transformer 惯例，频率从 1.0 到 1/10000 几何级数衰减
        # inv_freq shape: (dim,)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 1).float() / dim))
        
        # 2. 预计算位置相位 (Thetas)
        # t: (max_len,)
        t = torch.arange(max_seq_len).float()
        
        # freqs: (max_len, dim)
        # outer product: pos * freq
        freqs = torch.outer(t, inv_freq)
        
        # 3. 生成旋转因子 (Rotors)
        # rotor = e^{i * theta} = cos(theta) + i*sin(theta)
        # shape: (1, max_len, dim) 方便广播
        rotary_emb = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0)
        
        # 注册为 Buffer (不参与训练，但随模型移动到 GPU)
        self.register_buffer('rotary_emb', rotary_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Seq_Len, Dim) - Complex Tensor
        """
        seq_len = x.shape[1]
        
        # 截取对应长度的旋转因子
        # (1, Seq, Dim)
        rotors = self.rotary_emb[:, :seq_len, :]
        
        # 复数乘法：旋转相位
        # 这一步没有参数，也不需要 manual_backward 的特殊处理，
        # 因为它只是一个常数旋转，梯度流会自动穿过 (conj rotation)。
        return x * rotors

    def manual_backward(self, grad_output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        旋转的逆操作是反向旋转 (Conjugate)
        d(x * R)/dx = R
        grad_input = grad_output * R.conj()
        """
        seq_len = grad_output.shape[1]
        rotors = self.rotary_emb[:, :seq_len, :]
        
        # 反向旋转
        return grad_output * rotors.conj()
