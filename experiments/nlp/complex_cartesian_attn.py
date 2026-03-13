"""
complex_cartesian_attn.py

Python 接口: JIT 编译 CUDA kernel + autograd.Function + 集成到 Attention 模块

使用方式:
    from complex_cartesian_attn import CartesianDecomposedAttention
    # 自动检测 CUDA kernel 可用性, 不可用时 fallback 到 PyTorch 实现

验证数值等价性:
    python complex_cartesian_attn.py --verify
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional


# =====================================================================
#  JIT 编译 CUDA kernel
# =====================================================================

_cuda_module = None
_cuda_available = False


def _try_load_cuda():
    """尝试 JIT 编译 CUDA kernel。失败时返回 False。"""
    global _cuda_module, _cuda_available

    if _cuda_module is not None:
        return _cuda_available

    try:
        from torch.utils.cpp_extension import load

        # 定位 .cu 文件
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cu_path = os.path.join(this_dir, 'csrc', 'complex_cartesian_attn.cu')

        if not os.path.exists(cu_path):
            # 也尝试同级目录
            cu_path = os.path.join(this_dir, 'complex_cartesian_attn.cu')

        if not os.path.exists(cu_path):
            print(f"  [CUDA] .cu 文件未找到, 使用 PyTorch fallback")
            _cuda_available = False
            _cuda_module = False  # 标记为已尝试
            return False

        print(f"  [CUDA] JIT 编译 complex_cartesian_attn.cu ...")
        _cuda_module = load(
            name='complex_cartesian_attn',
            sources=[cu_path],
            verbose=False,
            extra_cuda_cflags=['-O3', '--use_fast_math', '-allow-unsupported-compiler'],
        )
        _cuda_available = True
        print(f"  [CUDA] 编译成功 ✓")
        return True

    except Exception as e:
        print(f"  [CUDA] 编译失败: {e}")
        print(f"  [CUDA] 使用 PyTorch fallback")
        _cuda_available = False
        _cuda_module = False
        return False


# =====================================================================
#  Autograd Function: 连接 CUDA kernel 和 PyTorch 自动微分
# =====================================================================


class ComplexCartesianAttnFn(Function):
    """
    自定义 autograd Function, 封装 CUDA forward/backward kernel。

    接收实数分量张量 (Qr, Qi, Kr, Ki, Vr, Vi), 返回 (Or, Oi)。
    内部:
      - forward: 调用 CUDA kernel, 保存 Lse 供 backward 使用
      - backward: 调用 CUDA backward kernel, 重计算 score (不物化 S×S)
    """

    @staticmethod
    def forward(ctx, Qr, Qi, Kr, Ki, Vr, Vi, scale):
        """
        Args:
            Qr, Qi, Kr, Ki, Vr, Vi: (N, S, Dh), float32, contiguous
            scale: float, = 1/√Dh
        Returns:
            Or, Oi: (N, S, Dh), float32
        """
        # 确保 contiguous
        Qr, Qi = Qr.contiguous(), Qi.contiguous()
        Kr, Ki = Kr.contiguous(), Ki.contiguous()
        Vr, Vi = Vr.contiguous(), Vi.contiguous()

        Or, Oi, Lse = _cuda_module.forward(Qr, Qi, Kr, Ki, Vr, Vi, scale)

        # 保存供 backward 使用 (不保存 S×S 矩阵!)
        ctx.save_for_backward(Qr, Qi, Kr, Ki, Vr, Vi, Or, Oi, Lse)
        ctx.scale = scale

        return Or, Oi

    @staticmethod
    def backward(ctx, dOr, dOi):
        """
        Args:
            dOr, dOi: (N, S, Dh), float32
        Returns:
            dQr, dQi, dKr, dKi, dVr, dVi, None (scale 不需要梯度)
        """
        Qr, Qi, Kr, Ki, Vr, Vi, Or, Oi, Lse = ctx.saved_tensors
        scale = ctx.scale

        dOr, dOi = dOr.contiguous(), dOi.contiguous()

        dQr, dQi, dKr, dKi, dVr, dVi = _cuda_module.backward(
            Qr, Qi, Kr, Ki, Vr, Vi, Or, Oi, Lse, dOr, dOi, scale)

        return dQr, dQi, dKr, dKi, dVr, dVi, None


def complex_cartesian_attn_cuda(Qr, Qi, Kr, Ki, Vr, Vi, scale):
    """调用 CUDA kernel 的便捷函数。"""
    return ComplexCartesianAttnFn.apply(Qr, Qi, Kr, Ki, Vr, Vi, scale)


# =====================================================================
#  PyTorch Fallback (当 CUDA kernel 不可用时)
# =====================================================================


def complex_cartesian_attn_pytorch(Q, K, V, scale):
    """
    PyTorch 参考实现 (cfloat 版本)。
    用于 CUDA kernel 不可用时的 fallback, 以及数值等价性验证。

    Args:
        Q, K, V: (N, S, Dh), cfloat
        scale: float
    Returns:
        O: (N, S, Dh), cfloat
    """
    scores = torch.matmul(Q, K.transpose(-2, -1).conj()) * scale
    attn_probs = torch.softmax(scores.real, dim=-1)
    phase_rotors = torch.polar(torch.ones_like(scores.imag), scores.imag)
    attn_weights = attn_probs * phase_rotors
    return torch.matmul(attn_weights, V)


# =====================================================================
#  工具函数
# =====================================================================

def complex_kaiming_init_(t):
    fan_in = t.shape[-1]
    std = 1.0 / math.sqrt(2.0 * fan_in)
    with torch.no_grad():
        t.copy_(torch.complex(torch.randn_like(t.real)*std, torch.randn_like(t.imag)*std))
    return t

def scaled_complex_kaiming_init_(t, scale=1.0):
    fan_in = t.shape[-1]
    std = (1.0/math.sqrt(2.0*fan_in)) * scale
    with torch.no_grad():
        t.copy_(torch.complex(torch.randn_like(t.real)*std, torch.randn_like(t.imag)*std))
    return t


class ComplexRotaryEmbedding(nn.Module):
    """RoPE 预计算缓存版。"""
    def __init__(self, d_head, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0/(base**(torch.arange(0,d_head).float()/d_head))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        angles = torch.outer(t, inv_freq)
        rotors = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer('rotors', rotors)
    def forward(self, z):
        if z.dim() == 4:
            return z * self.rotors[:z.shape[2]].unsqueeze(0).unsqueeze(0)
        else:
            return z * self.rotors[:z.shape[1]].unsqueeze(0)


# =====================================================================
#  CartesianDecomposedAttention — 自动选择 CUDA 或 PyTorch 路径
# =====================================================================


class CartesianDecomposedAttention(nn.Module):
    """
    笛卡尔正交分解注意力 — 自动选择最优实现。

    构造时尝试 JIT 编译 CUDA kernel:
      - 成功: forward 使用 CUDA kernel (O(S) 内存, 无 S×S 物化)
      - 失败: fallback 到 PyTorch cfloat 实现 (O(S²) 内存)

    两种路径数学完全等价, 参数格式统一为 cfloat。
    """

    def __init__(self, d_model, num_heads=4, num_layers=3):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        # QKV 融合 (cfloat 参数)
        self.wqkv = nn.Parameter(complex_kaiming_init_(
            torch.empty(3*d_model, d_model, dtype=torch.cfloat)))
        self.wo = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat),
            scale=1.0/math.sqrt(2.0*num_layers)))
        self.rotary = ComplexRotaryEmbedding(self.d_head)

        # 尝试加载 CUDA kernel
        self._use_cuda = _try_load_cuda()

    def forward(self, x):
        B, S, D = x.shape
        H, Dh = self.num_heads, self.d_head

        # QKV 融合投影
        qkv = F.linear(x, self.wqkv)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, S, H, Dh).transpose(1, 2)  # (B,H,S,Dh)
        K = K.view(B, S, H, Dh).transpose(1, 2)
        V = V.view(B, S, H, Dh).transpose(1, 2)

        # RoPE
        Q = self.rotary(Q)
        K = self.rotary(K)

        if self._use_cuda and x.is_cuda and Dh == 16:
            # ---- CUDA kernel 路径 ----
            # 将 (B,H,S,Dh) cfloat 转为 (B*H, S, Dh) 实数分量
            N = B * H
            Qr = Q.real.contiguous().view(N, S, Dh)
            Qi = Q.imag.contiguous().view(N, S, Dh)
            Kr = K.real.contiguous().view(N, S, Dh)
            Ki = K.imag.contiguous().view(N, S, Dh)
            Vr = V.real.contiguous().view(N, S, Dh)
            Vi = V.imag.contiguous().view(N, S, Dh)

            Or, Oi = complex_cartesian_attn_cuda(
                Qr, Qi, Kr, Ki, Vr, Vi, self.scale)

            # 重组为 (B,H,S,Dh) cfloat
            attn_out = torch.complex(
                Or.view(B, H, S, Dh),
                Oi.view(B, H, S, Dh))
        else:
            # ---- PyTorch fallback 路径 ----
            attn_out = complex_cartesian_attn_pytorch(Q, K, V, self.scale)

        # 合并头 + 输出投影
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return F.linear(attn_out, self.wo)


# =====================================================================
#  验证: CUDA kernel vs PyTorch 参考实现
# =====================================================================


def verify_numerical_equivalence(device='cuda', S=256, B=2, H=4, Dh=16):
    """验证 CUDA kernel 和 PyTorch 实现的数值等价性。"""
    print(f"\n{'='*60}")
    print(f"  数值等价性验证: S={S}, B={B}, H={H}, Dh={Dh}")
    print(f"{'='*60}")

    if not _try_load_cuda():
        print("  CUDA kernel 不可用, 跳过验证")
        return False

    torch.manual_seed(42)
    N = B * H

    # 随机复数输入
    Q = torch.complex(
        torch.randn(B, H, S, Dh, device=device),
        torch.randn(B, H, S, Dh, device=device)) * 0.1
    K = torch.complex(
        torch.randn(B, H, S, Dh, device=device),
        torch.randn(B, H, S, Dh, device=device)) * 0.1
    V = torch.complex(
        torch.randn(B, H, S, Dh, device=device),
        torch.randn(B, H, S, Dh, device=device)) * 0.1

    scale = 1.0 / math.sqrt(Dh)

    # ---- PyTorch 参考 ----
    Q_ref = Q.clone().requires_grad_(True)
    K_ref = K.clone().requires_grad_(True)
    V_ref = V.clone().requires_grad_(True)
    O_ref = complex_cartesian_attn_pytorch(Q_ref, K_ref, V_ref, scale)
    loss_ref = O_ref.abs().sum()
    loss_ref.backward()

    # ---- CUDA kernel ----
    Q_cuda = Q.clone()
    K_cuda = K.clone()
    V_cuda = V.clone()

    Qr = Q_cuda.real.contiguous().view(N, S, Dh).requires_grad_(True)
    Qi = Q_cuda.imag.contiguous().view(N, S, Dh).requires_grad_(True)
    Kr = K_cuda.real.contiguous().view(N, S, Dh).requires_grad_(True)
    Ki = K_cuda.imag.contiguous().view(N, S, Dh).requires_grad_(True)
    Vr = V_cuda.real.contiguous().view(N, S, Dh).requires_grad_(True)
    Vi = V_cuda.imag.contiguous().view(N, S, Dh).requires_grad_(True)

    Or, Oi = complex_cartesian_attn_cuda(Qr, Qi, Kr, Ki, Vr, Vi, scale)
    O_cuda = torch.complex(Or.view(B, H, S, Dh), Oi.view(B, H, S, Dh))
    loss_cuda = O_cuda.abs().sum()
    loss_cuda.backward()

    # ---- 对比 Forward ----
    fwd_diff = (O_cuda - O_ref).abs()
    fwd_max = fwd_diff.max().item()
    fwd_mean = fwd_diff.mean().item()
    fwd_ok = fwd_max < 1e-3

    print(f"\n  Forward:")
    print(f"    max |diff|  = {fwd_max:.2e}  {'✓' if fwd_ok else '✗ (> 1e-3)'}")
    print(f"    mean |diff| = {fwd_mean:.2e}")

    # ---- 对比 Backward ----
    def check_grad(name, grad_cuda, grad_ref_complex, dim_map):
        """对比实数梯度和复数参考梯度的对应分量。"""
        grad_ref = dim_map(grad_ref_complex)
        diff = (grad_cuda - grad_ref.contiguous().view_as(grad_cuda)).abs()
        mx = diff.max().item()
        mn = diff.mean().item()
        ok = mx < 1e-2
        print(f"    {name:10s}: max={mx:.2e}, mean={mn:.2e}  {'✓' if ok else '✗'}")
        return ok

    print(f"\n  Backward:")
    bwd_ok = True
    bwd_ok &= check_grad("dQr", Qr.grad, Q_ref.grad, lambda g: g.real)
    bwd_ok &= check_grad("dQi", Qi.grad, Q_ref.grad, lambda g: g.imag)
    bwd_ok &= check_grad("dKr", Kr.grad, K_ref.grad, lambda g: g.real)
    bwd_ok &= check_grad("dKi", Ki.grad, K_ref.grad, lambda g: g.imag)
    bwd_ok &= check_grad("dVr", Vr.grad, V_ref.grad, lambda g: g.real)
    bwd_ok &= check_grad("dVi", Vi.grad, V_ref.grad, lambda g: g.imag)

    # ---- 速度对比 ----
    print(f"\n  速度对比 (100 次 forward, S={S}):")

    # Warmup
    for _ in range(10):
        _ = complex_cartesian_attn_pytorch(Q, K, V, scale)
        Qr_t = Q.real.contiguous().view(N, S, Dh)
        Qi_t = Q.imag.contiguous().view(N, S, Dh)
        Kr_t = K.real.contiguous().view(N, S, Dh)
        Ki_t = K.imag.contiguous().view(N, S, Dh)
        Vr_t = V.real.contiguous().view(N, S, Dh)
        Vi_t = V.imag.contiguous().view(N, S, Dh)
        _ = _cuda_module.forward(Qr_t, Qi_t, Kr_t, Ki_t, Vr_t, Vi_t, scale)
    torch.cuda.synchronize()

    # PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = complex_cartesian_attn_pytorch(Q, K, V, scale)
    end.record()
    torch.cuda.synchronize()
    pytorch_ms = start.elapsed_time(end) / 100

    # CUDA kernel
    start.record()
    for _ in range(100):
        _ = _cuda_module.forward(Qr_t, Qi_t, Kr_t, Ki_t, Vr_t, Vi_t, scale)
    end.record()
    torch.cuda.synchronize()
    cuda_ms = start.elapsed_time(end) / 100

    speedup = pytorch_ms / cuda_ms
    print(f"    PyTorch cfloat: {pytorch_ms:.3f} ms")
    print(f"    CUDA kernel:    {cuda_ms:.3f} ms")
    print(f"    加速比:          {speedup:.2f}×")

    all_ok = fwd_ok and bwd_ok
    print(f"\n  {'全部通过 ✓' if all_ok else '存在误差 ✗ (检查上方标记)'}")
    return all_ok


# =====================================================================
#  命令行入口
# =====================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='运行数值等价性验证')
    parser.add_argument('--S', type=int, default=256)
    parser.add_argument('--B', type=int, default=2)
    args = parser.parse_args()

    if args.verify:
        verify_numerical_equivalence(S=args.S, B=args.B)
    else:
        print("使用 --verify 运行数值等价性验证")
        print("或在训练脚本中: from complex_cartesian_attn import CartesianDecomposedAttention")
