#!/usr/bin/env python3
"""
逐操作耗时对比: 复数模型 vs 实数基线

对 forward + backward 的每一步操作进行精确计时,
找出复数模型相对于实数基线的时间开销来源。

运行:
    python profile_complex_vs_real.py
    python profile_complex_vs_real.py --seq-len 256 --batch-size 64
"""

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


# =====================================================================
#  计时工具
# =====================================================================

class CudaTimer:
    """GPU 精确计时器, 使用 CUDA events 避免 CPU-GPU 同步偏差。"""
    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == 'cuda'
        self.records = {}

    @contextmanager
    def measure(self, name: str):
        if self.use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            yield
            elapsed_ms = (time.perf_counter() - t0) * 1000

        if name not in self.records:
            self.records[name] = []
        self.records[name].append(elapsed_ms)

    def summary(self, label: str):
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        total = 0
        for name, times in self.records.items():
            avg = sum(times) / len(times)
            total += avg
            print(f"  {name:40s}  {avg:8.3f} ms")
        print(f"  {'─' * 50}")
        print(f"  {'TOTAL':40s}  {total:8.3f} ms")
        return total


# =====================================================================
#  复数模型组件 (与 v9 一致, 用于逐步 profiling)
# =====================================================================

def complex_kaiming_init_(t):
    fan_in = t.shape[-1]
    std = 1.0 / math.sqrt(2.0 * fan_in)
    with torch.no_grad():
        t.copy_(torch.complex(torch.randn_like(t.real)*std, torch.randn_like(t.imag)*std))
    return t

def scaled_complex_kaiming_init_(t, scale=1.0):
    fan_in = t.shape[-1]
    std = (1.0 / math.sqrt(2.0 * fan_in)) * scale
    with torch.no_grad():
        t.copy_(torch.complex(torch.randn_like(t.real)*std, torch.randn_like(t.imag)*std))
    return t


def profile_complex_forward(B, S, D, H, ff_mult, n_layers, device, timer, n_warmup=3, n_repeat=10):
    """逐操作 profile 复数模型的一个 block 的 forward。"""
    Dh = D // H
    M = D * ff_mult

    # 创建参数
    wq = nn.Parameter(complex_kaiming_init_(torch.empty(D, D, dtype=torch.cfloat, device=device)))
    wk = nn.Parameter(complex_kaiming_init_(torch.empty(D, D, dtype=torch.cfloat, device=device)))
    wv = nn.Parameter(complex_kaiming_init_(torch.empty(D, D, dtype=torch.cfloat, device=device)))
    wo = nn.Parameter(scaled_complex_kaiming_init_(torch.empty(D, D, dtype=torch.cfloat, device=device), 1/math.sqrt(2*n_layers)))

    # QKV 融合版
    wqkv = nn.Parameter(complex_kaiming_init_(torch.empty(3*D, D, dtype=torch.cfloat, device=device)))

    w1 = nn.Parameter(complex_kaiming_init_(torch.empty(M, D, dtype=torch.cfloat, device=device)))
    w2 = nn.Parameter(scaled_complex_kaiming_init_(torch.empty(D, M, dtype=torch.cfloat, device=device), 1/math.sqrt(2*n_layers)))
    c = nn.Parameter(torch.polar(torch.ones(M, device=device), torch.rand(M, device=device)*2*math.pi))
    b = nn.Parameter(torch.zeros(M, device=device))
    norm_scale = nn.Parameter(torch.ones(D, device=device))

    # RoPE 缓存
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, Dh, device=device).float() / Dh))
    t_pos = torch.arange(S, device=device, dtype=inv_freq.dtype)
    angles = torch.outer(t_pos, inv_freq)
    rope_rotors = torch.polar(torch.ones_like(angles), angles).unsqueeze(0).unsqueeze(0)

    scale = 1.0 / math.sqrt(Dh)
    eps = 1e-6

    x = torch.complex(
        torch.randn(B, S, D, device=device) * 0.1,
        torch.randn(B, S, D, device=device) * 0.1)

    # Warmup
    for _ in range(n_warmup):
        Q = F.linear(x, wq)
        _ = torch.matmul(Q.view(B,S,H,Dh).transpose(1,2),
                          Q.view(B,S,H,Dh).transpose(1,2).transpose(-2,-1).conj())
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # ---- ATTENTION PROFILING ----
    for _ in range(n_repeat):
        # RMSNorm
        with timer.measure("[ℂ Attn] 1. ComplexRMSNorm"):
            rms = torch.sqrt(torch.mean(x.real**2 + x.imag**2, dim=-1, keepdim=True) + eps)
            z = (x / rms) * norm_scale

        # Q, K, V 分开投影 (v9 原版)
        with timer.measure("[ℂ Attn] 2a. Q/K/V 分开投影 (3×linear)"):
            Q = F.linear(z, wq).view(B, S, H, Dh).transpose(1, 2)
            K = F.linear(z, wk).view(B, S, H, Dh).transpose(1, 2)
            V = F.linear(z, wv).view(B, S, H, Dh).transpose(1, 2)

        # Q, K, V 融合投影 (优化版)
        with timer.measure("[ℂ Attn] 2b. QKV 融合投影 (1×linear)"):
            qkv = F.linear(z, wqkv)
            Q2, K2, V2 = qkv.chunk(3, dim=-1)
            Q2 = Q2.view(B, S, H, Dh).transpose(1, 2)
            K2 = K2.view(B, S, H, Dh).transpose(1, 2)
            V2 = V2.view(B, S, H, Dh).transpose(1, 2)

        # RoPE
        with timer.measure("[ℂ Attn] 3. RoPE (缓存版)"):
            Q = Q * rope_rotors
            K = K * rope_rotors

        with timer.measure("[ℂ Attn] 3-orig. RoPE (每次计算 polar)"):
            angles_new = torch.outer(t_pos, inv_freq)
            rotors_new = torch.polar(torch.ones_like(angles_new), angles_new)
            rotors_new = rotors_new.unsqueeze(0).unsqueeze(0)
            Q_alt = Q * rotors_new
            K_alt = K * rotors_new

        # Hermitian 内积
        with timer.measure("[ℂ Attn] 4. Q @ K^H (Hermitian matmul)"):
            scores = torch.matmul(Q, K.transpose(-2, -1).conj())

        with timer.measure("[ℂ Attn] 5. scores * scale"):
            scores = scores * scale

        # 笛卡尔分解
        with timer.measure("[ℂ Attn] 6. softmax(Re(S))"):
            attn_probs = torch.softmax(scores.real, dim=-1)

        with timer.measure("[ℂ Attn] 7a. torch.polar(1, Im(S))"):
            phase_rotors = torch.polar(torch.ones_like(scores.imag), scores.imag)

        with timer.measure("[ℂ Attn] 7b. torch.complex(cos,sin) 替代 polar"):
            im = scores.imag
            phase_alt = torch.complex(torch.cos(im), torch.sin(im))

        with timer.measure("[ℂ Attn] 8. attn_probs * phase_rotors"):
            attn_weights = attn_probs * phase_rotors

        with timer.measure("[ℂ Attn] 9. attn_weights @ V"):
            attn_out = torch.matmul(attn_weights, V)

        with timer.measure("[ℂ Attn] 10. reshape + output linear"):
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
            out = F.linear(attn_out, wo)

        # 残差
        with timer.measure("[ℂ Attn] 11. 残差连接"):
            x_after_attn = x + out

    # ---- DCU-FFN PROFILING ----
    for _ in range(n_repeat):
        with timer.measure("[ℂ DCU] 1. ComplexRMSNorm"):
            rms = torch.sqrt(torch.mean(x_after_attn.real**2 + x_after_attn.imag**2, dim=-1, keepdim=True) + eps)
            ffn_in = (x_after_attn / rms) * norm_scale

        with timer.measure("[ℂ DCU] 2. W1 linear (cfloat)"):
            h = F.linear(ffn_in, w1)

        with timer.measure("[ℂ DCU] 3a. c.conj()*h (cfloat 乘法)"):
            u = c.conj() * h

        with timer.measure("[ℂ DCU] 3b. 手工退相干 (避免 cfloat 中间张量)"):
            # s = c_r * h_r + c_i * h_i
            # t_raw = c_r * h_i - c_i * h_r
            s_alt = c.real * h.real + c.imag * h.imag
            t_raw_alt = c.real * h.imag - c.imag * h.real

        with timer.measure("[ℂ DCU] 4. 提取 s=Re(u), t=Im(u)/|c|"):
            s = u.real
            c_mag = torch.clamp(c.abs(), min=eps)
            t = u.imag / c_mag

        with timer.measure("[ℂ DCU] 5. gate_input + GELU + sigmoid*t"):
            gate_input = s - b
            o1 = F.gelu(gate_input)
            o2 = torch.sigmoid(gate_input) * t

        with timer.measure("[ℂ DCU] 6. torch.complex(o1, o2)"):
            h_measured = torch.complex(o1, o2)

        with timer.measure("[ℂ DCU] 7. W2 linear (cfloat)"):
            dcu_out = F.linear(h_measured, w2)

        with timer.measure("[ℂ DCU] 8. 残差连接"):
            x_out = x_after_attn + dcu_out


def profile_real_forward(B, S, d_real, H, ff_mult, device, timer, n_warmup=3, n_repeat=10):
    """逐操作 profile 实数基线的一个 block 的 forward。"""

    # 使用 nn.MultiheadAttention (与 RealByteMLM 一致)
    mha = nn.MultiheadAttention(d_real, H, batch_first=True, dropout=0.0).to(device)
    norm1 = nn.LayerNorm(d_real).to(device)
    norm2 = nn.LayerNorm(d_real).to(device)
    ff = nn.Sequential(
        nn.Linear(d_real, d_real * ff_mult),
        nn.GELU(),
        nn.Linear(d_real * ff_mult, d_real),
    ).to(device)

    x = torch.randn(B, S, d_real, device=device) * 0.1

    # Warmup
    for _ in range(n_warmup):
        h = norm1(x)
        h, _ = mha(h, h, h, need_weights=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    for _ in range(n_repeat):
        with timer.measure("[ℝ Attn] 1. LayerNorm"):
            h = norm1(x)

        with timer.measure("[ℝ Attn] 2. nn.MHA (QKV+SDPA+proj 融合)"):
            h, _ = mha(h, h, h, need_weights=False)

        with timer.measure("[ℝ Attn] 3. 残差连接"):
            x_after_attn = x + h

    for _ in range(n_repeat):
        with timer.measure("[ℝ FFN] 1. LayerNorm"):
            h = norm2(x_after_attn)

        with timer.measure("[ℝ FFN] 2. Linear(d→4d) + GELU + Linear(4d→d)"):
            h = ff(h)

        with timer.measure("[ℝ FFN] 3. 残差连接"):
            x_out = x_after_attn + h


def profile_backward(B, S, D_complex, D_real, H, ff_mult, n_layers, device, timer, n_warmup=2, n_repeat=5):
    """Profile 完整 forward+backward (单 block)。"""

    # --- 复数 ---
    from byte_mlm_v10_AB_fast import ComplexTransformerBlock, ComplexRMSNorm

    block_c = ComplexTransformerBlock(D_complex, H, ff_mult, n_layers).to(device)
    norm_c = ComplexRMSNorm(D_complex).to(device)

    x_c = torch.complex(
        torch.randn(B, S, D_complex, device=device, requires_grad=False) * 0.1,
        torch.randn(B, S, D_complex, device=device, requires_grad=False) * 0.1)
    x_c.requires_grad_(True)

    # warmup
    for _ in range(n_warmup):
        out = block_c(x_c)
        loss = out.abs().mean()
        loss.backward()
        block_c.zero_grad()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    for _ in range(n_repeat):
        with timer.measure("[ℂ Block] forward"):
            out = block_c(x_c)
        with timer.measure("[ℂ Block] backward"):
            loss = out.abs().mean()
            loss.backward()
            block_c.zero_grad()

    # --- 实数 ---
    mha = nn.MultiheadAttention(D_real, H, batch_first=True, dropout=0.0).to(device)
    norm1 = nn.LayerNorm(D_real).to(device)
    norm2 = nn.LayerNorm(D_real).to(device)
    ff = nn.Sequential(
        nn.Linear(D_real, D_real * ff_mult),
        nn.GELU(),
        nn.Linear(D_real * ff_mult, D_real),
    ).to(device)

    x_r = torch.randn(B, S, D_real, device=device, requires_grad=True)

    for _ in range(n_warmup):
        h = norm1(x_r); h, _ = mha(h, h, h, need_weights=False); h = x_r + h
        h2 = norm2(h); h2 = ff(h2); out = h + h2
        loss = out.mean(); loss.backward()
        mha.zero_grad(); ff.zero_grad()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    for _ in range(n_repeat):
        with timer.measure("[ℝ Block] forward"):
            h = norm1(x_r); h, _ = mha(h, h, h, need_weights=False); h = x_r + h
            h2 = norm2(h); h2 = ff(h2); out = h + h2
        with timer.measure("[ℝ Block] backward"):
            loss = out.mean(); loss.backward()
            mha.zero_grad(); ff.zero_grad()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    B, S = args.batch_size, args.seq_len
    D_complex, D_real = 64, 128
    H, ff_mult, n_layers = 4, 4, 3

    print(f"Device: {device}")
    if device.type == 'cuda':
        gpu = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu} (SM {cap[0]}.{cap[1]})")
        torch.set_float32_matmul_precision('high')
        print(f"TF32: {'启用' if cap[0] >= 8 else '不支持'}")

    print(f"\n配置: B={B}, S={S}, D_complex={D_complex}, D_real={D_real}, "
          f"H={H}, ff_mult={ff_mult}")

    # ---- Part 1: 逐操作 profiling ----
    print("\n" + "#" * 60)
    print("# Part 1: 复数模型 — 逐操作耗时")
    print("#" * 60)

    timer_c = CudaTimer(device)
    with torch.no_grad():
        profile_complex_forward(B, S, D_complex, H, ff_mult, n_layers, device, timer_c)
    total_c = timer_c.summary("复数模型 (1 Block, forward only, no grad)")

    print("\n" + "#" * 60)
    print("# Part 2: 实数基线 — 逐操作耗时")
    print("#" * 60)

    timer_r = CudaTimer(device)
    with torch.no_grad():
        profile_real_forward(B, S, D_real, H, ff_mult, device, timer_r)
    total_r = timer_r.summary("实数基线 (1 Block, forward only, no grad)")

    print(f"\n{'=' * 60}")
    print(f"  Forward 比值: ℂ/ℝ = {total_c/total_r:.2f}×")
    print(f"{'=' * 60}")

    # ---- Part 3: forward+backward block 级别 ----
    print("\n" + "#" * 60)
    print("# Part 3: 完整 forward+backward (1 Block)")
    print("#" * 60)

    timer_fb = CudaTimer(device)
    try:
        profile_backward(B, S, D_complex, D_real, H, ff_mult, n_layers, device, timer_fb)
        timer_fb.summary("Forward + Backward (1 Block)")
    except ImportError:
        print("  [跳过] 需要 byte_mlm_v10_AB_fast.py 在同一目录")
    except Exception as e:
        print(f"  [跳过] {e}")

    # ---- Part 4: 关键对比 ----
    print("\n" + "#" * 60)
    print("# Part 4: 关键对比和优化建议")
    print("#" * 60)

    print("""
  关键对比点:
    1. [ℂ Attn 2a] vs [ℂ Attn 2b]: QKV 融合的加速效果
    2. [ℂ Attn 3] vs [ℂ Attn 3-orig]: RoPE 缓存的加速效果
    3. [ℂ Attn 7a] vs [ℂ Attn 7b]: polar vs complex(cos,sin) 的对比
    4. [ℂ DCU 3a] vs [ℂ DCU 3b]: cfloat退相干 vs 手工退相干的对比
    5. [ℝ Attn 2] 一个操作包含了复数模型的 step 2-10 全部功能
       → 这是 nn.MHA 的 SDPA 融合优势, 无法用逐操作优化复现

  如果 [ℝ Attn 2] 远快于 [ℂ Attn 4+5+6+7+8+9] 的总和,
  则瓶颈在 SDPA 融合, 需要为复数 attention 实现自定义 CUDA kernel。

  如果两者接近, 则瓶颈在 cfloat 本身的开销, 可通过手工退相干等优化。
    """)


if __name__ == '__main__':
    main()
