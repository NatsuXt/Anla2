import torch
import torch.nn as nn
import math

def complex_normalize(z: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    复数归一化：保持相位不变，将模长归一化为 1。
    z_norm = z / |z|
    """
    return z / (torch.abs(z) + epsilon)

def complex_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    计算两个复数向量之间的欧几里得距离。
    d = ||z1 - z2||
    """
    return torch.abs(z1 - z2)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    如果未来需要 RoPE，这里保留接口。
    但在 Anla 中，我们通常直接在 Embedding 阶段处理相位。
    """
    pass

def get_phase_difference(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    计算相对相位差。
    返回值的范围在 [-pi, pi]
    """
    return torch.angle(z1 * torch.conj(z2))

def complex_kaiming_normal_(tensor_real, tensor_imag, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """
    [Anla Standard Init] 复数 Kaiming 初始化
    保持复数方差为 1，即 Var(Re) + Var(Im) = 1/fan_in
    """
    fan = nn.init._calculate_correct_fan(tensor_real, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    
    # 复数标准差分配给实部和虚部：sigma_real = sigma_imag = std / sqrt(2)
    with torch.no_grad():
        tensor_real.normal_(0, std / math.sqrt(2))
        tensor_imag.normal_(0, std / math.sqrt(2))

def create_complex_tensor(shape, requires_grad=True, device='cpu'):
    """创建复数张量容器 (辅助测试用)"""
    real = torch.empty(shape, device=device, requires_grad=requires_grad)
    imag = torch.empty(shape, device=device, requires_grad=requires_grad)
    complex_kaiming_normal_(real, imag)
    return torch.complex(real, imag)