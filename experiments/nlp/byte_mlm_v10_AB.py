#!/usr/bin/env python3
"""
Byte-level MLM v10-AB: 方案 A (非对称 DCU 初始化) + 方案 B (训练协议优化)

基于 v9 (CartesianAttn + PhaseFaithful-DCU-FFN) 的完整代码，新增两项改进:

【方案 A】DCU 非对称初始化
    问题: v9 中 c_m 的初始相位均匀随机，梯度下降在对称初始化下将 ~96% 的通道
          推向反对齐 (新颖性检测模式)，导致 FFN 有效容量被极度压缩。
          标称 256 通道中仅 ~10 个在任意时刻活跃。

    修改: 将 M 个 DCU 通道显式分为两组:
        - N 组 (Novelty): M_N 个通道, 初始化为反对齐 → 新颖性检测器
        - F 组 (Free):    M_F 个通道, 初始化为正对齐 → 自由非线性变换
    
    实现: 训练前对一个大 batch 做一次 forward pass (不反向传播),
          收集每个通道 h_m = W1^(m) · x 的 circular mean phase φ̄_m,
          N 组的 c_m 相位设为 φ̄_m + π (反对齐),
          F 组的 c_m 相位设为 φ̄_m     (正对齐).

    实验配置:
        - A-balanced:   M_N : M_F = 1:1 (128:128)
        - A-aggressive: M_N : M_F = 1:3 (64:192)

【方案 B】训练协议优化
    问题: v9 使用恒定 lr + 15K epochs, Dashboard 显示训练末端仍有上升趋势,
          训练未收敛。恒定 lr 导致模型在最优解附近振荡。

    修改:
        B-1: 余弦退火 (cosine annealing)
             lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*(t-T_w)/(T-T_w)))
             lr_max = 3e-4, lr_min = 1e-6, 线性 warmup 500 epochs
             总 epochs: 30000

        B-2: 梯度累积 (可选)
             grad_accum_steps = 2, 等效 batch_size 翻倍 (128)

运行方式:
    # 仅方案 B (训练协议优化, 原始初始化)
    python byte_mlm_v10_AB.py --mode B1 --seq-len 256

    # 方案 A-balanced + 方案 B-1
    python byte_mlm_v10_AB.py --mode A_balanced_B1 --seq-len 256

    # 方案 A-aggressive + 方案 B-1
    python byte_mlm_v10_AB.py --mode A_aggressive_B1 --seq-len 256

    # 方案 B-2 (余弦退火 + 梯度累积)
    python byte_mlm_v10_AB.py --mode B2 --seq-len 256

    # 全部实验一次性运行
    python byte_mlm_v10_AB.py --mode all --seq-len 256
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

# =====================================================================
#  路径设置
# =====================================================================

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# 尝试定位 Anla 项目根目录; 如果作为独立脚本运行则使用当前目录
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


def download_tiny_shakespeare(cache_dir: str) -> str:
    """下载 tiny_shakespeare 数据集到缓存目录。"""
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
    """
    字节级文本数据生成器。(与 v9 完全一致)

    将原始文本按字节 (0-255) 编码，生成 BERT 式的 MLM 训练批次:
    随机 mask 15% 的位置，模型预测被 mask 位置的原始字节值。
    """

    def __init__(self, data_path: str, seq_len: int, mask_id: int,
                 test_frac: float = 0.1, seed: int = 42):
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

    def _generate_batch(self, data: np.ndarray, batch_size: int,
                        mask_mode: str = 'bert', mask_prob: float = 0.15,
                        max_span: int = 5):
        """生成一个 MLM 训练批次。"""
        inp_list, tgt_list = [], []
        max_start = len(data) - self.seq_len
        for _ in range(batch_size):
            start = random.randint(0, max_start)
            seq = torch.tensor(
                data[start:start + self.seq_len].copy(), dtype=torch.long)
            inp, tgt = seq.clone(), torch.full_like(seq, -100)
            if mask_mode == 'bert':
                mask = torch.rand(self.seq_len) < mask_prob
                if not mask.any():
                    mask[random.randint(0, self.seq_len - 1)] = True
                inp[mask] = self.mask_id
                tgt[mask] = seq[mask]
            elif mask_mode == 'span':
                ml = random.randint(1, max_span)
                ms = random.randint(0, self.seq_len - ml)
                inp[ms:ms + ml] = self.mask_id
                tgt[ms:ms + ml] = seq[ms:ms + ml]
            inp_list.append(inp)
            tgt_list.append(tgt)
        return torch.stack(inp_list), torch.stack(tgt_list)

    def generate_train_batch(self, batch_size: int, **kw):
        return self._generate_batch(self.train_data, batch_size, **kw)

    def generate_test_batch(self, batch_size: int, **kw):
        return self._generate_batch(self.test_data, batch_size, **kw)


# =====================================================================
#  工具函数 (与 v9 完全一致)
# =====================================================================

def complex_kaiming_init_(tensor: torch.Tensor) -> torch.Tensor:
    """
    复数 Kaiming 初始化。

    对 ℂ^{out × in} 的权重矩阵，使得:
        Var(Re) = Var(Im) = 1 / (2 · fan_in)
    从而:
        E[|W_{ij}|²] = Var(Re) + Var(Im) = 1 / fan_in
    """
    fan_in = tensor.shape[-1]
    std = 1.0 / math.sqrt(2.0 * fan_in)
    with torch.no_grad():
        real = torch.randn_like(tensor.real) * std
        imag = torch.randn_like(tensor.imag) * std
        tensor.copy_(torch.complex(real, imag))
    return tensor


def scaled_complex_kaiming_init_(tensor: torch.Tensor,
                                 scale: float = 1.0) -> torch.Tensor:
    """
    复数 Kaiming 初始化 + 缩放因子。

    用于残差分支的最后一个复数线性层 (W_O, W₂_ℂ)，
    缩放因子 scale = 1/√(2L) 保证训练初期残差修正足够小。
    """
    fan_in = tensor.shape[-1]
    std = (1.0 / math.sqrt(2.0 * fan_in)) * scale
    with torch.no_grad():
        real = torch.randn_like(tensor.real) * std
        imag = torch.randn_like(tensor.imag) * std
        tensor.copy_(torch.complex(real, imag))
    return tensor


# =====================================================================
#  复数 Transformer 组件 (与 v9 完全一致, 除 DCU 初始化外)
# =====================================================================


class ComplexEmbedding(nn.Module):
    """复数嵌入层。(与 v9 完全一致)"""

    def __init__(self, num_embeddings: int, d_model: int):
        super().__init__()
        # 初始化: 实部和虚部独立 ~ N(0, 0.02)
        w = torch.complex(
            torch.randn(num_embeddings, d_model) * 0.02,
            torch.randn(num_embeddings, d_model) * 0.02,
        )
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class ComplexRMSNorm(nn.Module):
    """
    复数 RMSNorm。(与 v9 完全一致)

    对复数向量按模长的 RMS 做归一化:
        RMS = √(mean(|z_k|²) + ε)
        z_norm = (z / RMS) · γ
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))  # γ, 实数

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(
            torch.mean(z.real ** 2 + z.imag ** 2, dim=-1, keepdim=True)
            + self.eps
        )
        return (z / rms) * self.scale


class ComplexRotaryEmbedding(nn.Module):
    """
    复数旋转位置编码 (RoPE)。(与 v9 完全一致)

    利用复数乘法的旋转特性:
        q_k' = q_k · e^{i · pos · ω_k}
    """

    def __init__(self, d_head: int, max_seq_len: int = 512,
                 base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            seq_len = z.shape[2]
        else:
            seq_len = z.shape[1]

        t = torch.arange(seq_len, device=z.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq)  # (S, D_h)
        rotors = torch.polar(torch.ones_like(angles), angles)  # (S, D_h)

        if z.dim() == 4:
            return z * rotors.unsqueeze(0).unsqueeze(0)
        else:
            return z * rotors.unsqueeze(0)


# =====================================================================
#  【方案 B】笛卡尔正交分解注意力 (与 v9 完全一致, 无修改)
# =====================================================================


class CartesianDecomposedAttention(nn.Module):
    """
    笛卡尔正交分解注意力 — v9 核心创新之一。(与 v9 完全一致)

    v9 CartesianDecomposed:
        S = Q^H K / √D_h                     (复数 score)
        A = softmax(Re(S)) · exp(i · Im(S))   (实部选择 + 虚部旋转)
    """

    def __init__(self, d_model: int, num_heads: int = 4,
                 num_layers: int = 3):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V 投影: ℂ^D → ℂ^D
        self.wq = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wk = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wv = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))

        # 输出投影: ℂ^D → ℂ^D, 缩放初始化 1/√(2L)
        self.wo = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat),
            scale=1.0 / math.sqrt(2.0 * num_layers)))

        # RoPE
        self.rotary = ComplexRotaryEmbedding(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.num_heads, self.d_head

        # 1. 线性投影 + reshape 为多头
        Q = F.linear(x, self.wq).view(B, S, H, Dh).transpose(1, 2)
        K = F.linear(x, self.wk).view(B, S, H, Dh).transpose(1, 2)
        V = F.linear(x, self.wv).view(B, S, H, Dh).transpose(1, 2)

        # 2. 复数 RoPE
        Q = self.rotary(Q)
        K = self.rotary(K)

        # 3. Hermitian 内积: S = Q · K^H / √D_h
        scores = torch.matmul(Q, K.transpose(-2, -1).conj()) * (1.0 / math.sqrt(Dh))

        # 4. 笛卡尔正交分解
        attn_probs = torch.softmax(scores.real, dim=-1)  # (B,H,S,S) ℝ
        phase_rotors = torch.polar(
            torch.ones_like(scores.imag), scores.imag)    # (B,H,S,S) ℂ
        attn_weights = attn_probs * phase_rotors           # (B,H,S,S) ℂ

        # 5. 相干输运: O = A @ V
        attn_out = torch.matmul(attn_weights, V)  # (B,H,S,Dh) ℂ

        # 6. 合并头 + 输出投影
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        out = F.linear(attn_out, self.wo)

        return out


# =====================================================================
#  【方案 A 修改点】相位保真 DCU-FFN — 带非对称初始化支持
# =====================================================================


class PhaseFaithfulDCUFFN(nn.Module):
    """
    相位保真 DCU-FFN — v9 核心创新之二, 新增方案 A 非对称初始化。

    与 v9 的唯一区别: __init__ 中 self.c 的初始化方式可通过外部函数覆盖。
    forward 完全不变。

    信息流:
        x ∈ ℂ^D → W₁_ℂ → h ∈ ℂ^M → DCU(c*⊙h) → o¹,o² ∈ ℝ^M
        → ĥ = o¹ + i·o² ∈ ℂ^M → W₂_ℂ → out ∈ ℂ^D
    """

    def __init__(self, d_model: int, ff_mult: int = 4,
                 num_layers: int = 3):
        """
        Args:
            d_model: 复数维度 D
            ff_mult: FFN 扩展倍数 (M = ff_mult × D)
            num_layers: 总层数 L (用于 W₂_ℂ 的缩放初始化)
        """
        super().__init__()
        self.d_model = d_model
        self.ff_dim = d_model * ff_mult  # M = 4D
        self.eps = 1e-6

        # ---- Step 1 参数: 复数模式匹配 (线性扩展) ----
        # W₁ ∈ ℂ^{M × D}, 无偏置 (与 v9 完全一致)
        self.w1 = nn.Parameter(complex_kaiming_init_(
            torch.empty(self.ff_dim, d_model, dtype=torch.cfloat)))

        # ---- Step 2 参数: 本振 (Local Oscillator) ----
        # c ∈ ℂ^M: 每个通道的本振参数
        # 默认初始化与 v9 一致: 单位模长, 均匀随机相位
        # 方案 A 将在模型构建后通过 apply_asymmetric_init() 覆盖此初始化
        phases = torch.rand(self.ff_dim) * 2 * math.pi
        c_init = torch.polar(
            torch.ones(self.ff_dim),  # 模长 = 1
            phases                     # 相位 ∈ [0, 2π)
        )
        self.c = nn.Parameter(c_init)

        # ---- Step 3 参数: 门控阈值 ----
        # b ∈ ℝ^M (与 v9 完全一致)
        self.b = nn.Parameter(torch.zeros(self.ff_dim))

        # ---- Step 4 参数: 复数投影 (相干态制备) ----
        # W₂ ∈ ℂ^{D × M} (与 v9 完全一致)
        self.w2 = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, self.ff_dim, dtype=torch.cfloat),
            scale=1.0 / math.sqrt(2.0 * num_layers)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (与 v9 完全一致, 无任何修改):
            复数模式匹配 → 零差检测 → 双路门控 → 复数重组 → 复数投影

        Args:
            x: 复数输入 (经 RMSNorm 后), shape (B, S, D), dtype=cfloat
        Returns:
            复数输出, shape (B, S, D), dtype=cfloat
        """
        # Step 1: 复数模式匹配
        h = F.linear(x, self.w1)  # (B, S, M), ℂ

        # Step 2: 平衡零差检测
        u = self.c.conj() * h      # (B, S, M), ℂ
        s = u.real                  # 判据分量: Re(c* · h)
        c_mag = torch.clamp(self.c.abs(), min=self.eps)
        t = u.imag / c_mag         # 内容分量: Im(c* · h) / |c|

        # Step 3: 双路门控
        gate_input = s - self.b
        o1 = F.gelu(gate_input)           # 判据路径
        o2 = torch.sigmoid(gate_input) * t  # 内容路径

        # Step 4: 复数重组 + 复数投影
        h_measured = torch.complex(o1, o2)  # (B, S, M), ℂ
        out = F.linear(h_measured, self.w2)  # (B, S, D), ℂ

        return out


# =====================================================================
#  复数 Transformer Block (与 v9 完全一致)
# =====================================================================


class ComplexTransformerBlock(nn.Module):
    """
    复数 Transformer Block (Pre-Norm 残差结构)。

    每个子层遵循 Pre-Norm + 残差连接:
        x = x + SubLayer(RMSNorm(x))
    """

    def __init__(self, d_model: int, num_heads: int = 4,
                 ff_mult: int = 4, num_layers: int = 3):
        super().__init__()

        # Attention 子层: CartesianDecomposedAttention
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = CartesianDecomposedAttention(d_model, num_heads, num_layers)

        # FFN 子层: PhaseFaithfulDCUFFN
        self.norm2 = ComplexRMSNorm(d_model)
        self.ffn = PhaseFaithfulDCUFFN(d_model, ff_mult, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention 子层
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h

        # FFN 子层
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


# =====================================================================
#  完整模型: HoloDCU v10-AB Byte-level MLM
# =====================================================================


class HoloDCUByteMLM(nn.Module):
    """
    Phase-Faithful 复数 Transformer — Byte-level MLM。(与 v9 完全一致)

    完整架构:
        Input → ComplexEmbedding(ℂ^D)
        → [ComplexRMSNorm → CartesianAttn(ℂ) → residual
           → ComplexRMSNorm → PF-DCU-FFN(ℂ→ℝ→ℂ) → residual] × L
        → ComplexRMSNorm → Re(x · E^H) + b_cls → logits
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 num_heads: int = 4, num_blocks: int = 3,
                 ff_mult: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads, ff_mult, num_blocks)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.output_norm = ComplexRMSNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)           # (B, S, D), ℂ
        for block in self.blocks:
            z = block(z)                 # (B, S, D), ℂ
        z = self.output_norm(z)          # (B, S, D), ℂ

        # Weight Tying: logits_v = Re(E[v]^H · x)
        E = self.embedding.weight[:self.vocab_size]  # (V, D), ℂ
        logits = torch.real(torch.matmul(z, E.conj().T))
        logits = logits + self.output_bias

        return logits


# =====================================================================
#  【方案 A 核心】非对称 DCU 初始化
# =====================================================================


@torch.no_grad()
def apply_asymmetric_init(
    model: HoloDCUByteMLM,
    gen: TextByteGenerator,
    device: torch.device,
    novelty_ratio: float = 0.5,
    calibration_batch_size: int = 256,
    mask_prob: float = 0.15,
) -> Dict[str, Any]:
    """
    【方案 A】对已构建的模型执行非对称 DCU 初始化。

    算法:
        1. 对训练集的一个大 batch 做一次 forward pass (仅到 W₁·x)
        2. 收集每个 DCU 层中每个通道 h_m = W₁^(m) · x 的信号统计
        3. 计算每个通道的 circular mean phase φ̄_m
        4. 将前 M_N 个通道 (N 组) 的 c_m 相位设为 φ̄_m + π (反对齐)
        5. 将后 M_F 个通道 (F 组) 的 c_m 相位设为 φ̄_m     (正对齐)
        6. 所有 c_m 的模长保持为 1 (与 v9 初始化一致)

    Args:
        model: 已构建的 HoloDCUByteMLM 模型 (在 device 上)
        gen: 数据生成器
        device: 计算设备
        novelty_ratio: N 组占比, M_N = int(M * novelty_ratio)
                       0.5 → A-balanced (128:128)
                       0.25 → A-aggressive (64:192)
        calibration_batch_size: 校准用 batch 大小 (越大估计越准)
        mask_prob: 校准 batch 的 mask 概率 (与训练一致)

    Returns:
        初始化统计信息字典 (每层的分组数量、相位统计)
    """
    model.eval()

    # 生成校准 batch
    inp, _ = gen.generate_train_batch(
        calibration_batch_size, mask_mode='bert', mask_prob=mask_prob)
    inp = inp.to(device)

    # 获取 embedding 输出
    z = model.embedding(inp)  # (B, S, D), ℂ

    init_stats = {}

    for layer_idx, block in enumerate(model.blocks):
        # 获取该层 FFN 的输入 (经过 attention 和 norm 后)
        # 按 Pre-Norm 结构: FFN 输入 = norm2(x + attn(norm1(x)))
        with torch.no_grad():
            # 通过 attention 子层
            h_attn = block.norm1(z)
            h_attn = block.attn(h_attn)
            z_after_attn = z + h_attn

            # norm2 → 准备进入 FFN
            ffn_input = block.norm2(z_after_attn)  # (B, S, D), ℂ

            # 计算 h = W₁ · ffn_input  (DCU 的 Step 1)
            ffn = block.ffn
            h = F.linear(ffn_input, ffn.w1)  # (B, S, M), ℂ

        M = ffn.ff_dim
        M_N = int(M * novelty_ratio)  # N 组 (新颖性检测) 的通道数
        M_F = M - M_N                  # F 组 (自由变换) 的通道数

        # ---- 计算每个通道的 circular mean phase ----
        # h 的 shape: (B, S, M), ℂ
        # 对每个通道 m, 计算所有 token 的复数均值的辐角
        # circular mean = arg(mean(h_m / |h_m|))
        #               = arg(mean(exp(i * arg(h_m))))
        h_flat = h.reshape(-1, M)  # (B*S, M), ℂ
        h_mag = h_flat.abs().clamp(min=1e-8)
        h_unit = h_flat / h_mag     # 单位化, (B*S, M), ℂ

        # 复数均值 → 辐角 = circular mean phase
        h_mean = h_unit.mean(dim=0)  # (M,), ℂ
        phi_bar = torch.angle(h_mean)  # (M,), ℝ, 每个通道的典型相位

        # 计算 Rayleigh R (相位集中度) 用于诊断
        h_R = h_mean.abs()  # (M,), ℝ

        # ---- 设置 c_m 的新相位 ----
        new_c_phases = torch.zeros(M, device=device)

        # N 组 (前 M_N 个通道): 反对齐, c_m 相位 = φ̄_m + π
        new_c_phases[:M_N] = phi_bar[:M_N] + math.pi

        # F 组 (后 M_F 个通道): 正对齐, c_m 相位 = φ̄_m
        new_c_phases[M_N:] = phi_bar[M_N:]

        # 构建新的 c_m: 单位模长, 新相位
        new_c = torch.polar(torch.ones(M, device=device), new_c_phases)

        # 写入模型参数
        ffn.c.data.copy_(new_c)

        # ---- 记录统计信息 ----
        init_stats[f'layer_{layer_idx}'] = {
            'M': M,
            'M_N': M_N,
            'M_F': M_F,
            'novelty_ratio': novelty_ratio,
            'h_R_mean': h_R.mean().item(),
            'h_R_std': h_R.std().item(),
            'h_R_min': h_R.min().item(),
            'h_R_max': h_R.max().item(),
            # N 组的初始 cos(Δφ) 应该接近 -1 (反对齐)
            'N_group_cos_delta_mean': torch.cos(
                new_c_phases[:M_N] - phi_bar[:M_N]).mean().item(),
            # F 组的初始 cos(Δφ) 应该接近 +1 (正对齐)
            'F_group_cos_delta_mean': torch.cos(
                new_c_phases[M_N:] - phi_bar[M_N:]).mean().item(),
        }

        # ---- 传播到下一层 ----
        # 完成当前层的完整 forward (用于给下一层提供正确输入)
        with torch.no_grad():
            h_ffn = block.ffn(ffn_input)
            z = z_after_attn + h_ffn

        print(f"  [方案 A] Layer {layer_idx}: "
              f"M_N={M_N}, M_F={M_F}, "
              f"h_R_mean={h_R.mean().item():.4f}, "
              f"N_cos={init_stats[f'layer_{layer_idx}']['N_group_cos_delta_mean']:.4f}, "
              f"F_cos={init_stats[f'layer_{layer_idx}']['F_group_cos_delta_mean']:.4f}")

    model.train()
    return init_stats


# =====================================================================
#  实数基线 (与 v9 完全一致, 不做任何修改)
# =====================================================================


class RealRotaryEmbedding(nn.Module):
    """实数 RoPE。标准实现: cos/sin 对的旋转。"""

    def __init__(self, d_model: int, max_seq_len: int = 512,
                 base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        cos_f = freqs.cos().unsqueeze(0)
        sin_f = freqs.sin().unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.zeros_like(x)
        out[..., ::2] = x1 * cos_f - x2 * sin_f
        out[..., 1::2] = x1 * sin_f + x2 * cos_f
        return out


class RealByteMLM(nn.Module):
    """
    实数基线 Transformer — Byte-level MLM。(与 v9 完全一致)

    标准 Pre-Norm Transformer + GELU FFN + weight tying。
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 128,
                 num_heads: int = 4, num_blocks: int = 3,
                 ff_mult: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rotary = RealRotaryEmbedding(d_model, max_seq_len=512)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(
                    d_model, num_heads, batch_first=True, dropout=0.0),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * ff_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ff_mult, d_model)),
            }))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        h = self.rotary(h)
        for block in self.blocks:
            r = h
            h = block['norm1'](h)
            h, _ = block['attn'](h, h, h, need_weights=False)
            h = r + h
            r = h
            h = block['norm2'](h)
            h = block['ff'](h)
            h = r + h
        h = self.output_norm(h)
        return F.linear(h, self.embedding.weight[:self.vocab_size])


# =====================================================================
#  评估 (与 v9 完全一致)
# =====================================================================


@torch.no_grad()
def evaluate_model(model: nn.Module, gen: TextByteGenerator,
                   batch_size: int, device: torch.device,
                   num_batches: int = 10, mask_prob: float = 0.15
                   ) -> Tuple[float, float]:
    """
    统一评估: argmax accuracy + Cross-Entropy perplexity。
    """
    model.eval()
    total_correct, total_count, total_loss = 0, 0, 0.0
    for _ in range(num_batches):
        inp, tgt = gen.generate_test_batch(batch_size, mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)
        valid = (tgt != -100)
        if not valid.any():
            continue
        logits = model(inp)
        lv, tv = logits[valid], tgt[valid]
        total_loss += F.cross_entropy(lv, tv, reduction='sum').item()
        total_correct += (lv.argmax(-1) == tv).sum().item()
        total_count += tv.shape[0]
    acc = total_correct / max(total_count, 1)
    ppl = math.exp(min(total_loss / max(total_count, 1), 20.0))
    return acc, ppl


# =====================================================================
#  【方案 B 核心】训练函数 — 支持余弦退火 + 梯度累积
# =====================================================================


def make_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    lr_max: float,
    lr_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    【方案 B-1】创建带线性 warmup 的余弦退火 lr 调度器。

    调度公式:
        阶段 1 (0 ≤ t < T_w):
            lr(t) = lr_max * t / T_w                          (线性 warmup)

        阶段 2 (T_w ≤ t < T):
            lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*(t-T_w)/(T-T_w)))

    Args:
        optimizer: 优化器
        warmup_epochs: warmup 步数 T_w
        total_epochs: 总步数 T
        lr_max: warmup 后的峰值 lr
        lr_min: 余弦退火的最低 lr

    Returns:
        LambdaLR 调度器
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_epochs:
            # 线性 warmup: 从 0 线性增长到 lr_max
            return max(1, step) / warmup_epochs
        else:
            # 余弦退火: 从 lr_max 平滑衰减到 lr_min
            progress = (step - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            # 返回的是 lr_factor (乘以 optimizer 的 base lr)
            # base lr = lr_max, 所以 factor = (lr_min + cosine * (lr_max - lr_min)) / lr_max
            return (lr_min + cosine_factor * (lr_max - lr_min)) / lr_max

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model_v10(
    model: nn.Module,
    gen: TextByteGenerator,
    cfg: Dict[str, Any],
    device: torch.device,
    output_dir: str,
    model_name: str = 'complex',
    use_cosine_schedule: bool = True,
    grad_accum_steps: int = 1,
) -> Dict[str, Any]:
    """
    v10-AB 训练循环。

    与 v9 train_model 的区别:
        1. 【方案 B-1】支持余弦退火 lr 调度 (use_cosine_schedule=True)
        2. 【方案 B-2】支持梯度累积 (grad_accum_steps > 1)
        3. 记录额外的训练诊断信息 (lr 轨迹等)

    其他部分 (数据生成、loss 计算、评估、保存) 与 v9 完全一致。

    Args:
        model: 待训练模型
        gen: 数据生成器
        cfg: 配置字典
        device: 计算设备
        output_dir: 输出目录
        model_name: 模型名称标签
        use_cosine_schedule: 是否使用余弦退火 (方案 B-1); False 则退回 v9 的线性 warmup
        grad_accum_steps: 梯度累积步数 (方案 B-2); 1 = 无累积 (与 v9 一致)

    Returns:
        训练结果字典
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr_base = cfg['lr']
    warmup = cfg['warmup_epochs']
    wd = cfg['weight_decay']
    mask_prob = cfg['mask_prob']
    log_interval = cfg['log_interval']

    # ---- 优化器 (与 v9 一致) ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_base,
        weight_decay=wd, betas=(0.9, 0.999))

    # ---- 调度器: 方案 B-1 (余弦退火) vs v9 原始 (线性 warmup + 恒定) ----
    if use_cosine_schedule:
        scheduler = make_cosine_schedule(
            optimizer,
            warmup_epochs=warmup,
            total_epochs=epochs,
            lr_max=lr_base,
            lr_min=1e-6,
        )
        print(f"  [方案 B-1] 余弦退火: warmup={warmup}, "
              f"lr_max={lr_base}, lr_min=1e-6, total={epochs}")
    else:
        # v9 原始调度: 线性 warmup → 恒定 lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, max(1, s) / warmup))
        print(f"  [v9 原始] 线性 warmup={warmup} → 恒定 lr={lr_base}")

    if grad_accum_steps > 1:
        print(f"  [方案 B-2] 梯度累积: {grad_accum_steps} 步, "
              f"等效 batch_size={batch_size * grad_accum_steps}")

    history = {
        'epochs': [], 'loss': [], 'train_acc': [],
        'test_acc': [], 'test_ppl': [], 'lr': [],
    }

    best_train_acc = 0.0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()

        # ---- 梯度累积循环 ----
        # 当 grad_accum_steps=1 时, 与 v9 的行为完全一致
        accum_loss = 0.0
        accum_correct = 0
        accum_count = 0

        for accum_step in range(grad_accum_steps):
            inp, tgt = gen.generate_train_batch(
                batch_size, mask_mode='bert', mask_prob=mask_prob)
            inp, tgt = inp.to(device), tgt.to(device)
            valid = (tgt != -100)

            logits = model(inp)

            if valid.any():
                loss = F.cross_entropy(logits[valid], tgt[valid])
                # 梯度累积: loss 除以累积步数, 使总梯度等效于大 batch 的梯度
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                accum_loss += loss.item()
                with torch.no_grad():
                    accum_correct += (logits[valid].argmax(-1) == tgt[valid]).sum().item()
                    accum_count += tgt[valid].shape[0]
            else:
                # 极罕见情况: 没有 mask token
                loss_val = 0.0

        # ---- 参数更新 (每 grad_accum_steps 步一次) ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # ---- 日志记录 (与 v9 一致的间隔和格式) ----
        if epoch % log_interval == 0:
            avg_loss = accum_loss / grad_accum_steps
            tr_acc = accum_correct / max(accum_count, 1)

            test_acc, test_ppl = -1.0, -1.0
            if epoch % (log_interval * 5) == 0 or epoch == epochs - 1:
                test_acc, test_ppl = evaluate_model(
                    model, gen, batch_size, device, num_batches=5)

            history['epochs'].append(epoch)
            history['loss'].append(avg_loss)
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(test_acc)
            history['test_ppl'].append(test_ppl)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            if tr_acc > best_train_acc:
                best_train_acc = tr_acc
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'best.pth'))

            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""
            ppl_str = f" | PPL: {test_ppl:.1f}" if test_ppl >= 0 else ""
            lr_str = f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
            print(f"  [{model_name}] Ep {epoch:05d} | "
                  f"L: {avg_loss:.4f} | Tr: {tr_acc:.2%}"
                  f"{test_str}{ppl_str}{lr_str}")

    elapsed = time.time() - t_start
    final_acc, final_ppl = evaluate_model(
        model, gen, batch_size, device, num_batches=20)

    result = {
        'model': model_name,
        'best_train_acc': best_train_acc,
        'final_test_acc': final_acc,
        'final_test_ppl': final_ppl,
        'total_params': sum(p.numel() for p in model.parameters()),
        'total_real_params': sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in model.parameters()),
        'training_time_sec': elapsed,
        'history': history,
        'config': cfg,
        'use_cosine_schedule': use_cosine_schedule,
        'grad_accum_steps': grad_accum_steps,
    }
    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  [{model_name}] 完成: Train {best_train_acc:.2%}, "
          f"Test {final_acc:.2%}, PPL {final_ppl:.1f}, {elapsed:.0f}s")
    return result


# =====================================================================
#  单次实验运行器
# =====================================================================


def run_single_experiment(
    exp_name: str,
    gen: TextByteGenerator,
    cfg: Dict[str, Any],
    device: torch.device,
    base_output_dir: str,
    novelty_ratio: Optional[float] = None,
    use_cosine_schedule: bool = True,
    grad_accum_steps: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    运行一组 (复数 + 实数) 实验。

    Args:
        exp_name: 实验名称 (用作子目录名)
        gen: 数据生成器
        cfg: 配置字典
        device: 计算设备
        base_output_dir: 输出根目录
        novelty_ratio: 方案 A 的 N 组占比; None = 不使用方案 A (v9 原始初始化)
        use_cosine_schedule: 是否使用方案 B-1 (余弦退火)
        grad_accum_steps: 方案 B-2 的梯度累积步数

    Returns:
        {'complex': result_dict, 'real': result_dict}
    """
    exp_dir = os.path.join(base_output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    results = {}

    # ========== 复数模型 ==========
    print("\n" + "=" * 72)
    plan_a_str = (f"方案 A (novelty_ratio={novelty_ratio})"
                  if novelty_ratio is not None else "v9 原始初始化")
    plan_b_str = ("方案 B-1 (余弦退火)" if use_cosine_schedule
                  else "v9 原始调度")
    if grad_accum_steps > 1:
        plan_b_str += f" + B-2 (累积×{grad_accum_steps})"
    print(f"  [{exp_name}] 复数模型: {plan_a_str} + {plan_b_str}")
    print("=" * 72)

    d = cfg['d_model_complex']
    model_c = HoloDCUByteMLM(
        vocab_size=256, d_model=d,
        num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
        ff_mult=cfg['ff_mult'],
    ).to(device)

    # 参数量统计
    n_equiv = sum(p.numel() * (2 if p.is_complex() else 1)
                  for p in model_c.parameters())
    n_total = sum(p.numel() for p in model_c.parameters())
    print(f"  参数量: 总 {n_total:,}, 实数等效: {n_equiv:,}")

    # 方案 A: 非对称初始化 (如果指定)
    init_stats = None
    if novelty_ratio is not None:
        print(f"\n  执行方案 A 非对称初始化 (novelty_ratio={novelty_ratio})...")
        init_stats = apply_asymmetric_init(
            model_c, gen, device,
            novelty_ratio=novelty_ratio,
            calibration_batch_size=256,
            mask_prob=cfg['mask_prob'],
        )
        # 保存初始化统计
        with open(os.path.join(exp_dir, 'init_stats_complex.json'), 'w') as f:
            json.dump(init_stats, f, indent=2)
        print(f"  方案 A 初始化完成。统计已保存到 {exp_dir}/init_stats_complex.json")

    # 训练
    result_c = train_model_v10(
        model_c, gen, cfg, device,
        os.path.join(exp_dir, 'complex'),
        model_name=f'{exp_name}_complex',
        use_cosine_schedule=use_cosine_schedule,
        grad_accum_steps=grad_accum_steps,
    )
    if init_stats is not None:
        result_c['init_stats'] = init_stats
    results['complex'] = result_c

    # ========== 实数基线 ==========
    print("\n" + "=" * 72)
    print(f"  [{exp_name}] 实数基线: {plan_b_str}")
    print("=" * 72)

    d_r = cfg['d_model_real']
    model_r = RealByteMLM(
        vocab_size=256, d_model=d_r,
        num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
        ff_mult=cfg['ff_mult'],
    ).to(device)

    n_params_r = sum(p.numel() for p in model_r.parameters())
    print(f"  参数量: {n_params_r:,}")

    # 实数模型不需要方案 A, 但使用相同的方案 B 训练协议
    result_r = train_model_v10(
        model_r, gen, cfg, device,
        os.path.join(exp_dir, 'real'),
        model_name=f'{exp_name}_real',
        use_cosine_schedule=use_cosine_schedule,
        grad_accum_steps=grad_accum_steps,
    )
    results['real'] = result_r

    # ========== 对照总结 ==========
    print("\n" + "-" * 50)
    print(f"  [{exp_name}] 对照总结:")
    delta = result_c['final_test_acc'] - result_r['final_test_acc']
    print(f"    复数: Test {result_c['final_test_acc']:.2%}, "
          f"PPL {result_c['final_test_ppl']:.2f}, "
          f"Train(best) {result_c['best_train_acc']:.2%}")
    print(f"    实数: Test {result_r['final_test_acc']:.2%}, "
          f"PPL {result_r['final_test_ppl']:.2f}, "
          f"Train(best) {result_r['best_train_acc']:.2%}")
    print(f"    ℂ−ℝ 优势: {delta:+.2%}")
    print("-" * 50)

    return results


# =====================================================================
#  主入口
# =====================================================================


# 方案 B 的训练配置 (相对于 v9 DEFAULT_CFG 的变更)
V10_AB_CFG = {
    # 模型结构 (与 v9 完全一致, 不做任何修改)
    'd_model_complex': 64,
    'd_model_real': 128,
    'num_heads': 4,
    'num_blocks': 3,
    'ff_mult': 4,
    # 训练条件
    'seq_len': 256,           # 默认 seq256 (可通过命令行覆盖)
    'batch_size': 64,         # 与 v9 一致
    'lr': 3e-4,               # 与 v9 一致 (余弦退火的峰值 lr)
    'weight_decay': 0.01,     # 与 v9 一致
    'epochs': 30000,          # 【方案 B 修改】v9 = 15000, v10 = 30000
    'warmup_epochs': 500,     # 与 v9 一致
    'mask_prob': 0.15,        # 与 v9 一致
    'log_interval': 200,      # 与 v9 一致
}


def main():
    parser = argparse.ArgumentParser(
        description="Byte-level MLM v10-AB: "
                    "方案 A (非对称 DCU 初始化) + 方案 B (训练协议优化)")

    parser.add_argument(
        '--mode', type=str, default='all',
        choices=[
            'B1',                # 仅方案 B-1 (余弦退火), v9 原始初始化
            'B2',                # 方案 B-2 (余弦退火 + 梯度累积)
            'A_balanced_B1',     # 方案 A (1:1) + B-1
            'A_aggressive_B1',   # 方案 A (1:3) + B-1
            'all',               # 全部实验
        ],
        help="实验模式")

    parser.add_argument('--data-path', type=str, default=None,
                        help="数据文件路径 (默认自动下载 tiny_shakespeare)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="覆盖训练 epoch 数")
    parser.add_argument('--seq-len', type=int, default=None,
                        help="覆盖序列长度")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="覆盖 batch 大小")
    parser.add_argument('--lr', type=float, default=None,
                        help="覆盖学习率")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="输出根目录")
    parser.add_argument('--device', type=str, default='auto',
                        help="计算设备 (auto/cpu/cuda/mps)")

    args = parser.parse_args()

    # ---- 设备选择 ----
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- 配置 ----
    cfg = V10_AB_CFG.copy()
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.seq_len is not None:
        cfg['seq_len'] = args.seq_len
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['lr'] = args.lr

    # ---- 输出目录 ----
    if args.output_dir is not None:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(
            _ANLA_ROOT, 'Logs',
            f'nlp_byte_mlm_v10AB_seq{cfg["seq_len"]}')

    print(f"\n配置:")
    print(f"  seq_len={cfg['seq_len']}, batch_size={cfg['batch_size']}, "
          f"epochs={cfg['epochs']}")
    print(f"  lr={cfg['lr']}, warmup={cfg['warmup_epochs']}, "
          f"wd={cfg['weight_decay']}")
    print(f"  输出目录: {base_output_dir}")

    # ---- 数据加载 ----
    if args.data_path is None:
        data_path = download_tiny_shakespeare(
            os.path.join(_ANLA_ROOT, 'data'))
    else:
        data_path = args.data_path

    gen = TextByteGenerator(
        data_path, cfg['seq_len'], mask_id=256, test_frac=0.1)

    # ---- 定义实验列表 ----
    experiments: List[Dict[str, Any]] = []

    if args.mode in ('B1', 'all'):
        experiments.append({
            'name': 'B1_cosine_only',
            'novelty_ratio': None,         # v9 原始初始化
            'use_cosine_schedule': True,    # 方案 B-1
            'grad_accum_steps': 1,          # 无梯度累积
        })

    if args.mode in ('B2', 'all'):
        experiments.append({
            'name': 'B2_cosine_grad_accum',
            'novelty_ratio': None,          # v9 原始初始化
            'use_cosine_schedule': True,     # 方案 B-1
            'grad_accum_steps': 2,           # 方案 B-2: 梯度累积×2
        })

    if args.mode in ('A_balanced_B1', 'all'):
        experiments.append({
            'name': 'A_balanced_B1',
            'novelty_ratio': 0.5,           # 方案 A: N:F = 1:1 (128:128)
            'use_cosine_schedule': True,     # 方案 B-1
            'grad_accum_steps': 1,
        })

    if args.mode in ('A_aggressive_B1', 'all'):
        experiments.append({
            'name': 'A_aggressive_B1',
            'novelty_ratio': 0.25,          # 方案 A: N:F = 1:3 (64:192)
            'use_cosine_schedule': True,     # 方案 B-1
            'grad_accum_steps': 1,
        })

    # ---- 运行实验 ----
    all_results = {}
    for exp in experiments:
        print(f"\n{'#' * 72}")
        print(f"# 实验: {exp['name']}")
        print(f"{'#' * 72}")

        results = run_single_experiment(
            exp_name=exp['name'],
            gen=gen,
            cfg=cfg,
            device=device,
            base_output_dir=base_output_dir,
            novelty_ratio=exp['novelty_ratio'],
            use_cosine_schedule=exp['use_cosine_schedule'],
            grad_accum_steps=exp['grad_accum_steps'],
        )
        all_results[exp['name']] = results

    # ---- 最终汇总 ----
    print(f"\n\n{'=' * 72}")
    print("  v10-AB 全部实验汇总")
    print(f"{'=' * 72}")
    print(f"  seq_len={cfg['seq_len']}, epochs={cfg['epochs']}")
    print()

    # 表头
    print(f"  {'实验名':25s} | {'ℂ Test':>8s} | {'ℂ PPL':>6s} | "
          f"{'ℂ Best Tr':>9s} | {'ℝ Test':>8s} | {'ℝ PPL':>6s} | "
          f"{'ℂ−ℝ':>6s} | {'ℂ Time':>7s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-"
          f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")

    for exp_name, res in all_results.items():
        rc = res['complex']
        rr = res['real']
        delta = rc['final_test_acc'] - rr['final_test_acc']
        print(f"  {exp_name:25s} | "
              f"{rc['final_test_acc']:7.2%} | "
              f"{rc['final_test_ppl']:6.2f} | "
              f"{rc['best_train_acc']:8.2%} | "
              f"{rr['final_test_acc']:7.2%} | "
              f"{rr['final_test_ppl']:6.2f} | "
              f"{delta:+5.2%} | "
              f"{rc['training_time_sec']:6.0f}s")

    # 保存汇总
    summary_path = os.path.join(base_output_dir, 'summary.json')
    summary = {}
    for exp_name, res in all_results.items():
        summary[exp_name] = {
            'complex_test_acc': res['complex']['final_test_acc'],
            'complex_test_ppl': res['complex']['final_test_ppl'],
            'complex_best_train_acc': res['complex']['best_train_acc'],
            'real_test_acc': res['real']['final_test_acc'],
            'real_test_ppl': res['real']['final_test_ppl'],
            'real_best_train_acc': res['real']['best_train_acc'],
            'delta_acc': (res['complex']['final_test_acc']
                          - res['real']['final_test_acc']),
            'complex_time': res['complex']['training_time_sec'],
            'real_time': res['real']['training_time_sec'],
        }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  汇总已保存到: {summary_path}")
    print(f"  详细日志: {base_output_dir}/*/{{complex,real}}/training_log.json")


if __name__ == '__main__':
    main()
