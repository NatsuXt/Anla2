#!/usr/bin/env python3
"""
Byte-level MLM v10-AB-fast: 针对 profile 热点的精确加速

基于 v10-AB 完整功能, 针对 profile 确认的两个热点做精确优化:

【热点 1: Attention score 张量 — 占复数模型 47% 耗时】
    问题: (B,H,S,S) cfloat score 张量 = 128MB, 每次操作 (scale, .real, .imag,
          polar, 乘法) 都是 128MB 的全量读写, 5 次操作 = 640MB 内存流量。
          而实数基线用 SDPA 从不物化此张量。

    优化: 计算 Re(Q@K^H) 和 Im(Q@K^H) 为两个独立的实数张量 (各 64MB)。
          所有后续操作 (softmax, cos, sin, 乘法) 在实数张量上进行。
          - 消除 cfloat 交错布局的缓存低效
          - softmax 直接操作实数张量, 不需要 .real 提取
          - cos/sin 直接输出实数, 不需要 torch.polar 创建复数

【热点 2: DCU 退相干 — 占复数模型 22% 耗时】
    问题: u = c.conj() * h 产生 cfloat 中间张量, 然后 u.real / u.imag
          做 strided 提取 + c.abs() + clamp + 除法 = 4 次独立 kernel。

    优化: 直接用 c 和 h 的 real/imag 分量计算 s 和 t, 避免创建 cfloat u:
          s = cr*hr + ci*hi          (1 次 kernel: fused multiply-add)
          t = (cr*hi - ci*hr) * c_inv (1 次 kernel, c_inv 预计算)

【不变的部分】
    - 所有参数 (Wqkv, Wo, W1, W2, c) 保持 cfloat — 投影 matmul 在小矩阵上快
    - QKV 融合投影、RoPE 缓存、TF32 — 上一版已验证有效的优化保留
    - 方案 A (非对称初始化) + 方案 B (余弦退火) 完整保留
    - 数学上与 v9 完全等价

运行 (与 v10-AB 命令行完全兼容):
    python byte_mlm_v10_AB_fast.py --mode B1 --seq-len 256
    python byte_mlm_v10_AB_fast.py --mode A_aggressive_B1 --seq-len 256
    python byte_mlm_v10_AB_fast.py --mode all --seq-len 256
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
if not os.path.isdir(os.path.join(_ANLA_ROOT, 'experiments')):
    _ANLA_ROOT = _FILE_DIR
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)


# =====================================================================
#  数据加载 (与 v9 完全一致)
# =====================================================================

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

def download_tiny_shakespeare(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tiny_shakespeare.txt")
    if os.path.exists(path):
        print(f"  [数据] 使用缓存: {path}")
        return path
    print(f"  [数据] 下载 tiny_shakespeare...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    print(f"  [数据] 完成: {os.path.getsize(path):,} bytes -> {path}")
    return path

class TextByteGenerator:
    def __init__(self, data_path, seq_len, mask_id, test_frac=0.1, seed=42):
        self.seq_len = seq_len
        self.mask_id = mask_id
        with open(data_path, 'rb') as f:
            raw = f.read()
        self.data = np.frombuffer(raw, dtype=np.uint8)
        split = int(len(self.data) * (1.0 - test_frac))
        self.train_data = self.data[:split]
        self.test_data = self.data[split:]
        print(f"  [数据] 总: {len(self.data):,}, "
              f"训练: {len(self.train_data):,}, 测试: {len(self.test_data):,}, "
              f"唯一 byte: {len(np.unique(self.data))}/256")

    def _generate_batch(self, data, batch_size, mask_mode='bert',
                        mask_prob=0.15, max_span=5):
        inp_list, tgt_list = [], []
        max_start = len(data) - self.seq_len
        for _ in range(batch_size):
            start = random.randint(0, max_start)
            seq = torch.tensor(data[start:start+self.seq_len].copy(), dtype=torch.long)
            inp, tgt = seq.clone(), torch.full_like(seq, -100)
            if mask_mode == 'bert':
                mask = torch.rand(self.seq_len) < mask_prob
                if not mask.any():
                    mask[random.randint(0, self.seq_len-1)] = True
                inp[mask] = self.mask_id
                tgt[mask] = seq[mask]
            elif mask_mode == 'span':
                ml = random.randint(1, max_span)
                ms = random.randint(0, self.seq_len-ml)
                inp[ms:ms+ml] = self.mask_id
                tgt[ms:ms+ml] = seq[ms:ms+ml]
            inp_list.append(inp)
            tgt_list.append(tgt)
        return torch.stack(inp_list), torch.stack(tgt_list)

    def generate_train_batch(self, batch_size, **kw):
        return self._generate_batch(self.train_data, batch_size, **kw)
    def generate_test_batch(self, batch_size, **kw):
        return self._generate_batch(self.test_data, batch_size, **kw)


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
    std = (1.0 / math.sqrt(2.0 * fan_in)) * scale
    with torch.no_grad():
        t.copy_(torch.complex(torch.randn_like(t.real)*std, torch.randn_like(t.imag)*std))
    return t


# =====================================================================
#  复数 Transformer 组件
# =====================================================================

class ComplexEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model):
        super().__init__()
        w = torch.complex(
            torch.randn(num_embeddings, d_model) * 0.02,
            torch.randn(num_embeddings, d_model) * 0.02)
        self.weight = nn.Parameter(w)
    def forward(self, x):
        return self.weight[x]


class ComplexRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
    def forward(self, z):
        # 【优化】直接从 cfloat 的 real/imag view 计算, 避免 z.abs()**2
        rms = torch.sqrt(
            torch.mean(z.real**2 + z.imag**2, dim=-1, keepdim=True) + self.eps)
        return (z / rms) * self.scale


class ComplexRotaryEmbedding(nn.Module):
    """RoPE — 预计算旋转因子缓存。"""
    def __init__(self, d_head, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head).float() / d_head))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        angles = torch.outer(t, inv_freq)
        rotors = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer('rotors', rotors)
    def forward(self, z):
        if z.dim() == 4:
            return z * self.rotors[:z.shape[2]].unsqueeze(0).unsqueeze(0)
        else:
            return z * self.rotors[:z.shape[1]].unsqueeze(0)


class CartesianDecomposedAttention(nn.Module):
    """
    笛卡尔正交分解注意力 — CUDA forward + PyTorch backward 混合策略。

    Forward: CUDA kernel (O(S) 内存, 5.93× 加速, 无 S×S 物化)
    Backward: PyTorch 实数 score 路径 (autograd 自动微分, 无 atomicAdd)

    CUDA kernel 的 backward 中 atomicAdd 在高竞争下性能退化,
    因此 backward 回退到 PyTorch: 重计算 forward 再让 autograd 处理梯度。
    """
    _cuda_mod = None
    _cuda_ready = None

    @classmethod
    def _ensure_cuda(cls):
        if cls._cuda_ready is not None:
            return cls._cuda_ready
        try:
            from torch.utils.cpp_extension import load
            this_dir = os.path.dirname(os.path.abspath(__file__))
            cu_path = os.path.join(this_dir, 'csrc', 'complex_cartesian_attn.cu')
            if not os.path.exists(cu_path):
                cu_path = os.path.join(this_dir, 'complex_cartesian_attn.cu')
            if not os.path.exists(cu_path):
                print("  [CUDA] .cu 文件未找到, 使用 PyTorch fallback")
                cls._cuda_ready = False
                return False
            print("  [CUDA] JIT 编译 complex_cartesian_attn.cu ...")
            cls._cuda_mod = load(
                name='complex_cartesian_attn',
                sources=[cu_path],
                verbose=False,
                extra_cuda_cflags=['-O3', '--use_fast_math', '-allow-unsupported-compiler'],
            )
            cls._cuda_ready = True
            print("  [CUDA] 编译成功 ✓ — forward 使用 Complex FlashAttention kernel")
            return True
        except Exception as e:
            print(f"  [CUDA] 编译失败: {str(e)[:120]}")
            print(f"  [CUDA] 使用 PyTorch fallback")
            cls._cuda_ready = False
            return False

    def __init__(self, d_model, num_heads=4, num_layers=3):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.wqkv = nn.Parameter(complex_kaiming_init_(
            torch.empty(3 * d_model, d_model, dtype=torch.cfloat)))
        self.wo = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat),
            scale=1.0 / math.sqrt(2.0 * num_layers)))
        self.rotary = ComplexRotaryEmbedding(self.d_head)
        self._ensure_cuda()

    @staticmethod
    def _pytorch_attn_core(Q, K, V, scale):
        """
        PyTorch 实数 score 路径 — 用于 backward 的 autograd 重计算。

        所有操作都走 PyTorch autograd 图, backward 由 PyTorch 自动处理。
        实数 score 张量 (非 cfloat) 已经比原版快, 同时避免了 CUDA atomicAdd。
        """
        Qr = Q.real.contiguous()
        Qi = Q.imag.contiguous()
        Kr = K.real.contiguous()
        Ki = K.imag.contiguous()
        KrT = Kr.transpose(-2, -1)
        KiT = Ki.transpose(-2, -1)

        re_scores = (torch.matmul(Qr, KrT) + torch.matmul(Qi, KiT)) * scale
        im_scores = (torch.matmul(Qi, KrT) - torch.matmul(Qr, KiT)) * scale

        attn_probs = torch.softmax(re_scores, dim=-1)
        cos_im = torch.cos(im_scores)
        sin_im = torch.sin(im_scores)
        aw_cos = attn_probs * cos_im
        aw_sin = attn_probs * sin_im

        Vr = V.real.contiguous()
        Vi = V.imag.contiguous()
        out_r = torch.matmul(aw_cos, Vr) - torch.matmul(aw_sin, Vi)
        out_i = torch.matmul(aw_cos, Vi) + torch.matmul(aw_sin, Vr)
        return torch.complex(out_r, out_i)

    def forward(self, x):
        B, S, D = x.shape
        H, Dh = self.num_heads, self.d_head

        # QKV 融合投影
        qkv = F.linear(x, self.wqkv)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, S, H, Dh).transpose(1, 2)
        K = K.view(B, S, H, Dh).transpose(1, 2)
        V = V.view(B, S, H, Dh).transpose(1, 2)

        # RoPE
        Q = self.rotary(Q)
        K = self.rotary(K)

        if self._cuda_ready and x.is_cuda and Dh == 16 and not torch.is_grad_enabled():
            # ---- 推理/评估: CUDA kernel (最快, O(S) 内存) ----
            N = B * H
            Qr = Q.real.contiguous().view(N, S, Dh)
            Qi = Q.imag.contiguous().view(N, S, Dh)
            Kr = K.real.contiguous().view(N, S, Dh)
            Ki = K.imag.contiguous().view(N, S, Dh)
            Vr = V.real.contiguous().view(N, S, Dh)
            Vi = V.imag.contiguous().view(N, S, Dh)
            Or, Oi, _ = self._cuda_mod.forward(Qr, Qi, Kr, Ki, Vr, Vi, self.scale)
            attn_out = torch.complex(Or.view(B, H, S, Dh), Oi.view(B, H, S, Dh))
        else:
            # ---- 训练: PyTorch 实数 score 路径 (autograd backward, 无 atomicAdd) ----
            attn_out = self._pytorch_attn_core(Q, K, V, self.scale)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return F.linear(attn_out, self.wo)


class PhaseFaithfulDCUFFN(nn.Module):
    """
    相位保真 DCU-FFN — 退相干热点优化。

    【核心优化: 避免 cfloat 中间张量 u】
    旧 (v9): u = c.conj() * h → s = u.real → t = u.imag / |c|
        → 创建 cfloat u, 然后 strided 提取 .real/.imag, 4 次独立 kernel

    新: 直接从 c 和 h 的 real/imag 分量计算 s 和 t
        cr, ci = c.real, c.imag  (参数, 已缓存)
        hr, hi = h.real, h.imag  (一次 contiguous 拷贝)
        s = cr * hr + ci * hi    (1 次 fused kernel)
        t = (cr * hi - ci * hr) * c_mag_inv  (1 次 fused kernel)

    c_mag_inv = 1/|c| 每次 forward 更新一次 (c 是 Parameter, 每步变化)。
    """
    def __init__(self, d_model, ff_mult=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = d_model * ff_mult
        self.eps = 1e-6

        self.w1 = nn.Parameter(complex_kaiming_init_(
            torch.empty(self.ff_dim, d_model, dtype=torch.cfloat)))

        phases = torch.rand(self.ff_dim) * 2 * math.pi
        self.c = nn.Parameter(torch.polar(torch.ones(self.ff_dim), phases))

        self.b = nn.Parameter(torch.zeros(self.ff_dim))

        self.w2 = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, self.ff_dim, dtype=torch.cfloat),
            scale=1.0 / math.sqrt(2.0 * num_layers)))

    def forward(self, x):
        # Step 1: W1 投影 (cfloat matmul, 快)
        h = F.linear(x, self.w1)  # (B, S, M), cfloat

        # Step 2:【核心优化】直接实数退相干, 避免 cfloat 中间张量
        # 提取 c 和 h 的 real/imag (c 是 1D 参数, h 需要 contiguous)
        cr = self.c.real  # (M,), 已连续
        ci = self.c.imag  # (M,), 已连续
        hr = h.real       # (B,S,M), cfloat 的 real view
        hi = h.imag       # (B,S,M), cfloat 的 imag view

        # 判据分量: s = Re(c* · h) = cr*hr + ci*hi
        s = cr * hr + ci * hi  # (B,S,M), 实数

        # 内容分量: t = Im(c* · h) / |c| = (cr*hi - ci*hr) / |c|
        c_mag_inv = 1.0 / torch.clamp(self.c.abs(), min=self.eps)  # (M,)
        t = (cr * hi - ci * hr) * c_mag_inv  # (B,S,M), 实数

        # Step 3: 双路门控 (全实数, 与 v9 一致)
        gate_input = s - self.b
        o1 = F.gelu(gate_input)
        o2 = torch.sigmoid(gate_input) * t

        # Step 4: 复数重组 + W2 投影 (cfloat matmul, 快)
        return F.linear(torch.complex(o1, o2), self.w2)


class ComplexTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, ff_mult=4, num_layers=3):
        super().__init__()
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = CartesianDecomposedAttention(d_model, num_heads, num_layers)
        self.norm2 = ComplexRMSNorm(d_model)
        self.ffn = PhaseFaithfulDCUFFN(d_model, ff_mult, num_layers)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HoloDCUByteMLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, num_heads=4,
                 num_blocks=3, ff_mult=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads, ff_mult, num_blocks)
            for _ in range(num_blocks)])
        self.output_norm = ComplexRMSNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        z = self.embedding(x)
        for block in self.blocks:
            z = block(z)
        z = self.output_norm(z)
        E = self.embedding.weight[:self.vocab_size]
        logits = torch.real(torch.matmul(z, E.conj().T))
        return logits + self.output_bias


# =====================================================================
#  方案 A: 非对称 DCU 初始化
# =====================================================================

@torch.no_grad()
def apply_asymmetric_init(model, gen, device, novelty_ratio=0.5,
                          calibration_batch_size=256, mask_prob=0.15):
    model.eval()
    inp, _ = gen.generate_train_batch(
        calibration_batch_size, mask_mode='bert', mask_prob=mask_prob)
    inp = inp.to(device)
    z = model.embedding(inp)
    init_stats = {}
    for layer_idx, block in enumerate(model.blocks):
        with torch.no_grad():
            h_attn = block.attn(block.norm1(z))
            z_after = z + h_attn
            ffn_input = block.norm2(z_after)
            h = F.linear(ffn_input, block.ffn.w1)
        M = block.ffn.ff_dim
        M_N = int(M * novelty_ratio)
        h_flat = h.reshape(-1, M)
        h_unit = h_flat / h_flat.abs().clamp(min=1e-8)
        h_mean = h_unit.mean(dim=0)
        phi_bar = torch.angle(h_mean)
        h_R = h_mean.abs()
        new_phases = torch.zeros(M, device=device)
        new_phases[:M_N] = phi_bar[:M_N] + math.pi
        new_phases[M_N:] = phi_bar[M_N:]
        block.ffn.c.data.copy_(torch.polar(torch.ones(M, device=device), new_phases))
        init_stats[f'layer_{layer_idx}'] = {
            'M': M, 'M_N': M_N, 'M_F': M - M_N,
            'novelty_ratio': novelty_ratio,
            'h_R_mean': h_R.mean().item(),
            'N_cos': torch.cos(new_phases[:M_N] - phi_bar[:M_N]).mean().item(),
            'F_cos': torch.cos(new_phases[M_N:] - phi_bar[M_N:]).mean().item(),
        }
        with torch.no_grad():
            z = z_after + block.ffn(ffn_input)
        print(f"  [方案 A] L{layer_idx}: M_N={M_N}, M_F={M-M_N}, "
              f"h_R={h_R.mean().item():.4f}")
    model.train()
    return init_stats


# =====================================================================
#  实数基线 (与 v9 完全一致)
# =====================================================================

class RealRotaryEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, base=10000.0):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1.0/(base**(torch.arange(0,d_model,2).float()/d_model))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x):
        S = x.shape[1]
        t = torch.arange(S, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        c, s = freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)
        x1, x2 = x[...,::2], x[...,1::2]
        out = torch.zeros_like(x)
        out[...,::2] = x1*c - x2*s
        out[...,1::2] = x1*s + x2*c
        return out

class RealByteMLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, num_heads=4,
                 num_blocks=3, ff_mult=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size+1, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rotary = RealRotaryEmbedding(d_model, max_seq_len=512)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model,num_heads,batch_first=True,dropout=0.0),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(nn.Linear(d_model,d_model*ff_mult),nn.GELU(),
                                     nn.Linear(d_model*ff_mult,d_model)),
            }))
        self.output_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.rotary(self.embedding(x))
        for block in self.blocks:
            r = h; h = block['norm1'](h); h,_ = block['attn'](h,h,h,need_weights=False); h = r+h
            r = h; h = block['norm2'](h); h = block['ff'](h); h = r+h
        return F.linear(self.output_norm(h), self.embedding.weight[:self.vocab_size])


# =====================================================================
#  评估 + 训练 + 实验运行器 (与 v10-AB 一致)
# =====================================================================

@torch.no_grad()
def evaluate_model(model, gen, batch_size, device, num_batches=10, mask_prob=0.15):
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    for _ in range(num_batches):
        inp, tgt = gen.generate_test_batch(batch_size, mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)
        valid = (tgt != -100)
        if not valid.any(): continue
        logits = model(inp)
        lv, tv = logits[valid], tgt[valid]
        loss_sum += F.cross_entropy(lv, tv, reduction='sum').item()
        correct += (lv.argmax(-1) == tv).sum().item()
        count += tv.shape[0]
    return correct/max(count,1), math.exp(min(loss_sum/max(count,1), 20.0))

def make_cosine_schedule(optimizer, warmup, total, lr_max, lr_min=1e-6):
    def lr_lambda(step):
        if step < warmup: return max(1,step)/warmup
        p = (step-warmup)/max(1,total-warmup)
        return (lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*p)))/lr_max
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model_v10(model, gen, cfg, device, output_dir, model_name='complex',
                    use_cosine_schedule=True, grad_accum_steps=1):
    os.makedirs(output_dir, exist_ok=True)
    epochs,bs,lr_base = cfg['epochs'],cfg['batch_size'],cfg['lr']
    warmup,wd = cfg['warmup_epochs'],cfg['weight_decay']
    mask_prob,log_int = cfg['mask_prob'],cfg['log_interval']

    opt = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=wd, betas=(0.9,0.999))
    if use_cosine_schedule:
        sched = make_cosine_schedule(opt, warmup, epochs, lr_base, 1e-6)
        print(f"  [B-1] 余弦退火: warmup={warmup}, total={epochs}")
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0,max(1,s)/warmup))
    if grad_accum_steps > 1:
        print(f"  [B-2] 梯度累积×{grad_accum_steps}")

    hist = {'epochs':[],'loss':[],'train_acc':[],'test_acc':[],'test_ppl':[],'lr':[]}
    best_tr = 0.0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        al, ac, an = 0.0, 0, 0
        for _ in range(grad_accum_steps):
            inp, tgt = gen.generate_train_batch(bs, mask_mode='bert', mask_prob=mask_prob)
            inp, tgt = inp.to(device), tgt.to(device)
            valid = (tgt != -100)
            logits = model(inp)
            if valid.any():
                loss = F.cross_entropy(logits[valid], tgt[valid])
                (loss/grad_accum_steps).backward()
                al += loss.item()
                with torch.no_grad():
                    ac += (logits[valid].argmax(-1)==tgt[valid]).sum().item()
                    an += tgt[valid].shape[0]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(); sched.step()

        if ep % log_int == 0:
            avg_l = al/grad_accum_steps
            tr = ac/max(an,1)
            ta, tp = -1.0, -1.0
            if ep % (log_int*5)==0 or ep==epochs-1:
                ta, tp = evaluate_model(model, gen, bs, device, num_batches=5)
            hist['epochs'].append(ep); hist['loss'].append(avg_l)
            hist['train_acc'].append(tr); hist['test_acc'].append(ta)
            hist['test_ppl'].append(tp); hist['lr'].append(opt.param_groups[0]['lr'])
            if tr > best_tr:
                best_tr = tr
                torch.save(model.state_dict(), os.path.join(output_dir,'best.pth'))
            ts = f" | Test: {ta:.2%}" if ta>=0 else ""
            ps = f" | PPL: {tp:.1f}" if tp>=0 else ""
            print(f"  [{model_name}] Ep {ep:05d} | L: {avg_l:.4f} | Tr: {tr:.2%}{ts}{ps}"
                  f" | LR: {opt.param_groups[0]['lr']:.2e}")

    elapsed = time.time()-t0
    fa, fp = evaluate_model(model, gen, bs, device, num_batches=20)
    result = {
        'model':model_name, 'best_train_acc':best_tr,
        'final_test_acc':fa, 'final_test_ppl':fp,
        'total_params':sum(p.numel() for p in model.parameters()),
        'total_real_params':sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
        'training_time_sec':elapsed, 'history':hist, 'config':cfg,
    }
    with open(os.path.join(output_dir,'training_log.json'),'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  [{model_name}] 完成: Train {best_tr:.2%}, Test {fa:.2%}, PPL {fp:.1f}, {elapsed:.0f}s")
    return result

def run_single_experiment(exp_name, gen, cfg, device, base_dir,
                          novelty_ratio=None, use_cosine=True, grad_accum=1):
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    results = {}

    # 复数
    print("\n" + "="*72)
    pa = f"方案A(ratio={novelty_ratio})" if novelty_ratio is not None else "原始初始化"
    pb = "B-1(余弦退火)" if use_cosine else "原始调度"
    if grad_accum > 1: pb += f"+B-2(累积×{grad_accum})"
    print(f"  [{exp_name}] 复数: {pa} + {pb}")
    print("="*72)
    d = cfg['d_model_complex']
    mc = HoloDCUByteMLM(vocab_size=256, d_model=d,
        num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
        ff_mult=cfg['ff_mult']).to(device)
    ne = sum(p.numel()*(2 if p.is_complex() else 1) for p in mc.parameters())
    print(f"  参数量: 实数等效 {ne:,}")
    if novelty_ratio is not None:
        st = apply_asymmetric_init(mc, gen, device, novelty_ratio=novelty_ratio,
                                   calibration_batch_size=256, mask_prob=cfg['mask_prob'])
        with open(os.path.join(exp_dir,'init_stats.json'),'w') as f: json.dump(st,f,indent=2)
    rc = train_model_v10(mc, gen, cfg, device, os.path.join(exp_dir,'complex'),
                         f'{exp_name}_complex', use_cosine, grad_accum)
    results['complex'] = rc

    # 实数
    print("\n"+"="*72)
    print(f"  [{exp_name}] 实数基线: {pb}")
    print("="*72)
    mr = RealByteMLM(vocab_size=256, d_model=cfg['d_model_real'],
        num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
        ff_mult=cfg['ff_mult']).to(device)
    print(f"  参数量: {sum(p.numel() for p in mr.parameters()):,}")
    rr = train_model_v10(mr, gen, cfg, device, os.path.join(exp_dir,'real'),
                         f'{exp_name}_real', use_cosine, grad_accum)
    results['real'] = rr
    delta = rc['final_test_acc']-rr['final_test_acc']
    print(f"\n  [{exp_name}] ℂ:{rc['final_test_acc']:.2%} ℝ:{rr['final_test_acc']:.2%} "
          f"Δ:{delta:+.2%} ℂ:{rc['training_time_sec']:.0f}s ℝ:{rr['training_time_sec']:.0f}s")
    return results


# =====================================================================
#  主入口
# =====================================================================

V10_AB_CFG = {
    'd_model_complex':64, 'd_model_real':128,
    'num_heads':4, 'num_blocks':3, 'ff_mult':4,
    'seq_len':256, 'batch_size':64,
    'lr':3e-4, 'weight_decay':0.01,
    'epochs':30000, 'warmup_epochs':500,
    'mask_prob':0.15, 'log_interval':200,
}

def main():
    parser = argparse.ArgumentParser(description="v10-AB-fast: 针对 profile 热点的精确加速")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['B1','B2','A_balanced_B1','A_aggressive_B1','all'])
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available()
                              else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        gpu = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu} (SM {cap[0]}.{cap[1]})")
        print(f"  [TF32] {'已启用' if cap[0]>=8 else '不支持 (需SM≥8.0)'}")

    cfg = V10_AB_CFG.copy()
    if args.epochs: cfg['epochs'] = args.epochs
    if args.seq_len: cfg['seq_len'] = args.seq_len
    if args.batch_size: cfg['batch_size'] = args.batch_size
    if args.lr: cfg['lr'] = args.lr

    base_dir = args.output_dir or os.path.join(
        _ANLA_ROOT, 'Logs', f'nlp_byte_mlm_v10AB_fast_seq{cfg["seq_len"]}')
    print(f"\n配置: seq_len={cfg['seq_len']}, batch={cfg['batch_size']}, epochs={cfg['epochs']}")
    print(f"输出: {base_dir}")

    data_path = args.data_path or download_tiny_shakespeare(os.path.join(_ANLA_ROOT,'data'))
    gen = TextByteGenerator(data_path, cfg['seq_len'], mask_id=256, test_frac=0.1)

    exps = []
    if args.mode in ('B1','all'):
        exps.append(dict(name='B1_cosine_only', nr=None, cs=True, ga=1))
    if args.mode in ('B2','all'):
        exps.append(dict(name='B2_cosine_grad_accum', nr=None, cs=True, ga=2))
    if args.mode in ('A_balanced_B1','all'):
        exps.append(dict(name='A_balanced_B1', nr=0.5, cs=True, ga=1))
    if args.mode in ('A_aggressive_B1','all'):
        exps.append(dict(name='A_aggressive_B1', nr=0.25, cs=True, ga=1))

    all_res = {}
    for e in exps:
        print(f"\n{'#'*72}\n# 实验: {e['name']}\n{'#'*72}")
        all_res[e['name']] = run_single_experiment(
            e['name'], gen, cfg, device, base_dir,
            novelty_ratio=e['nr'], use_cosine=e['cs'], grad_accum=e['ga'])

    # 汇总
    print(f"\n\n{'='*72}\n  v10-AB-fast 实验汇总\n{'='*72}")
    print(f"  {'实验':25s} | {'ℂTest':>7s} | {'ℂPPL':>5s} | {'ℝTest':>7s} | {'Δ':>6s} | {'ℂTime':>6s} | {'ℝTime':>6s}")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for n,r in all_res.items():
        c,rr=r['complex'],r['real']
        d=c['final_test_acc']-rr['final_test_acc']
        print(f"  {n:25s} | {c['final_test_acc']:6.2%} | {c['final_test_ppl']:5.2f} | "
              f"{rr['final_test_acc']:6.2%} | {d:+5.2%} | {c['training_time_sec']:5.0f}s | "
              f"{rr['training_time_sec']:5.0f}s")

    sm_path = os.path.join(base_dir, 'summary.json')
    sm = {n: {'c_acc':r['complex']['final_test_acc'],'c_ppl':r['complex']['final_test_ppl'],
              'c_best_tr':r['complex']['best_train_acc'],
              'r_acc':r['real']['final_test_acc'],'r_ppl':r['real']['final_test_ppl'],
              'delta':r['complex']['final_test_acc']-r['real']['final_test_acc'],
              'c_time':r['complex']['training_time_sec'],
              'r_time':r['real']['training_time_sec']}
         for n,r in all_res.items()}
    os.makedirs(base_dir, exist_ok=True)
    with open(sm_path,'w') as f: json.dump(sm,f,indent=2)
    print(f"\n  汇总: {sm_path}")

if __name__ == '__main__':
    main()
