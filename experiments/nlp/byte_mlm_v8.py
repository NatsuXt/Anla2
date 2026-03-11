"""
保存位置: Anla/experiments/nlp/byte_mlm_v8.py

Byte-Level MLM v8 — Holographic Attention + DCU-FFN 复数 Transformer
=========================================================================

v8 架构 (基于量子测量理论的 Hybrid ℂ-ℝ 架构):

    信息流拓扑:
        Input → ComplexEmbedding(ℂ^D)
        → [ComplexRMSNorm → HoloAttn(ℂ) → residual
           → ComplexRMSNorm → DCU-FFN(ℂ→ℝ→ℂ) → residual] × L
        → ComplexRMSNorm → Re(x · E^H) + b_cls → logits

    核心创新:

    1. HolographicAttention (autograd 版):
       复数 Q^H K → MagPhaseSoftmax (softmax(|S|) · S/|S|) → 复数 A @ V
       注意力权重是复数——模长编码关注强度，相位编码关注方式。
       不同于 v7 的 Re(Q^H K) 实数 softmax，完整保留相位信息。

    2. DCU-FFN (Decoherence Coding Unit 融合的 FFN):
       基于平衡零差检测 (balanced homodyne detection) 原理:
       · 复数线性扩展 W₁_ℂ: 信号场
       · 可学习本振 c ∈ ℂ^M: 定义测量方向和增益
       · 平衡检测: s = Re(c* ⊙ h) (判据分量), t = Im(c* ⊙ h)/|c| (内容分量)
       · 双路门控: GELU(s-b) (判据路径), σ(s-b)·t (内容路径)
       · 实数投影 W₂_ℝ → reshape → 复数残差

       物理含义: 每层执行一次 "量子干涉 → 退相干测量 → 经典处理 → 量子反馈"

    3. Weight Tying (Hermitian 内积):
       logits_v = Re(E[v]^H · x) + b_cls[v]
       复数 embedding 作为输出层的本振——与 DCU 的零差检测框架一致。

    对照基线:
        实数 Transformer (与 v7 完全一致的实数基线)

用法:
    python -m Anla.experiments.nlp.byte_mlm_v8 --model complex
    python -m Anla.experiments.nlp.byte_mlm_v8 --model real
    python -m Anla.experiments.nlp.byte_mlm_v8 --model both
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)


# =====================================================================
#  数据加载 (复用 v7 的 TextByteGenerator)
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
    字节级文本数据生成器。

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
#  工具函数
# =====================================================================

def complex_kaiming_init_(tensor: torch.Tensor) -> torch.Tensor:
    """
    复数 Kaiming 初始化。

    对 ℂ^{out × in} 的权重矩阵，使得:
        Var(Re) = Var(Im) = 1 / (2 · fan_in)
    从而:
        E[|W_{ij}|²] = Var(Re) + Var(Im) = 1 / fan_in

    这保证了前向传播中信号方差不随层数衰减或爆炸。
    """
    fan_in = tensor.shape[-1]
    std = 1.0 / math.sqrt(2.0 * fan_in)  # 每个分量的标准差
    with torch.no_grad():
        real = torch.randn_like(tensor.real) * std
        imag = torch.randn_like(tensor.imag) * std
        tensor.copy_(torch.complex(real, imag))
    return tensor


def scaled_real_kaiming_init_(tensor: torch.Tensor,
                              scale: float = 1.0) -> torch.Tensor:
    """
    实数 Kaiming 初始化 + 缩放因子。

    用于残差分支的最后一个线性层 (W_O, W_2)，
    缩放因子 scale = 1/√(2L) 保证训练初期残差修正足够小，
    等价于 GPT-2 / DeepNet 的残差缩放初始化策略。
    """
    fan_in = tensor.shape[-1]
    std = (1.0 / math.sqrt(fan_in)) * scale
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def scaled_complex_kaiming_init_(tensor: torch.Tensor,
                                 scale: float = 1.0) -> torch.Tensor:
    """
    复数 Kaiming 初始化 + 缩放因子。

    用于注意力输出投影 W_O 的缩放初始化。
    """
    fan_in = tensor.shape[-1]
    std = (1.0 / math.sqrt(2.0 * fan_in)) * scale
    with torch.no_grad():
        real = torch.randn_like(tensor.real) * std
        imag = torch.randn_like(tensor.imag) * std
        tensor.copy_(torch.complex(real, imag))
    return tensor


# =====================================================================
#  复数 Transformer 组件 (全部使用 PyTorch autograd)
# =====================================================================


class ComplexEmbedding(nn.Module):
    """
    复数嵌入层。

    与 v7 的 AGComplexEmbedding 不同，本版本:
    - 不做单位圆归一化 (允许模长自由学习)
    - 使用 complex_kaiming_normal_ 初始化

    这支持「自发对称性破缺」: 高频 token 可以学到更大的 embedding 模长，
    在 weight-tied 输出层中获得更高的 logit 基线。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # 初始化: Var(Re) + Var(Im) = 1/D
        w = torch.empty(num_embeddings, embedding_dim, dtype=torch.cfloat)
        complex_kaiming_init_(w)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token ids, shape (B, S), dtype=long
        Returns:
            复数 embedding, shape (B, S, D), dtype=cfloat
        """
        return self.weight[x]


class ComplexRMSNorm(nn.Module):
    """
    复数 RMSNorm。

    对复数向量按模长的 RMS 做归一化:
        RMS = √(mean(|z_k|²) + ε)
        z_norm = (z / RMS) · γ

    γ ∈ ℝ^D 是逐维度的可学习缩放因子 (实数标量缩放复数向量，
    只改变模长不改变相位)。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))  # γ, 实数

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 复数张量, shape (..., D), dtype=cfloat
        Returns:
            归一化后的复数张量, shape 同输入
        """
        # |z_k|² = z_k.real² + z_k.imag²
        rms = torch.sqrt(
            torch.mean(z.real ** 2 + z.imag ** 2, dim=-1, keepdim=True)
            + self.eps
        )
        return (z / rms) * self.scale


class ComplexRotaryEmbedding(nn.Module):
    """
    复数旋转位置编码 (RoPE)。

    利用复数乘法的旋转特性，对 Q, K 的每个维度施加位置相关的旋转:
        q_k' = q_k · e^{i · pos · ω_k}

    其中 ω_k = 1/10000^(k/D_h) 是每个维度的旋转频率。

    复数 RoPE 比实数 RoPE (cos/sin 对) 更自然:
    一次复数乘法同时完成了实数版本中 4 次乘法 + 2 次加法的工作。
    """

    def __init__(self, d_head: int, max_seq_len: int = 512,
                 base: float = 10000.0):
        super().__init__()
        # 频率: ω_k = 1/base^(k/d_head), k = 0, 1, ..., d_head-1
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: shape (B, H, S, D_h) 或 (B, S, D_h), dtype=cfloat
        Returns:
            旋转后的复数张量, shape 同输入
        """
        # 确定序列维度的位置
        if z.dim() == 4:
            # (B, H, S, D_h) — 多头注意力内部的 Q/K
            seq_len = z.shape[2]
        else:
            # (B, S, D_h) — 单头或 embedding 后
            seq_len = z.shape[1]

        # 位置索引
        t = torch.arange(seq_len, device=z.device, dtype=self.inv_freq.dtype)
        # 旋转角度: θ_{pos, k} = pos · ω_k
        angles = torch.outer(t, self.inv_freq)  # (S, D_h)
        # 旋转因子: e^{iθ}
        rotors = torch.polar(torch.ones_like(angles), angles)  # (S, D_h)

        if z.dim() == 4:
            # (B, H, S, D_h) × (1, 1, S, D_h) 广播
            return z * rotors.unsqueeze(0).unsqueeze(0)
        else:
            # (B, S, D_h) × (1, S, D_h) 广播
            return z * rotors.unsqueeze(0)


class HolographicAttention(nn.Module):
    """
    全息共振注意力 (Holographic Resonance Attention) — autograd 版本。

    与 v7 的 AGComplexAttention 的关键区别:
    - v7: score = Re(Q^H K) → 实数 softmax → 实数权重 × 复数 V
      (在注意力分数阶段就丢弃了相位信息)
    - v8: score = Q^H K → MagPhaseSoftmax → 复数权重 × 复数 V
      (完整保留相位，注意力权重是复数)

    MagPhaseSoftmax:
        A_{ij} = softmax_j(|S_{ij}|) · S_{ij} / |S_{ij}|_ε

    物理含义:
    - 模长经 softmax 归一化 → 选择最相干的信号源
    - 相位完整透传 → Value 在加权叠加前被旋转，形成全息干涉图样
    """

    def __init__(self, d_model: int, num_heads: int = 4,
                 num_layers: int = 3):
        """
        Args:
            d_model: 复数维度 D
            num_heads: 注意力头数 H
            num_layers: 总层数 L (用于 W_O 的缩放初始化)
        """
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V 投影: ℂ^D → ℂ^D (复数线性层)
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

        # RoPE (每个头独立的旋转编码)
        self.rotary = ComplexRotaryEmbedding(self.d_head)

        # MagPhaseSoftmax 的数值保护参数
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 投影 → RoPE → 共轭干涉 → MagPhaseSoftmax → 相干输运 → 输出投影

        Args:
            x: 输入, shape (B, S, D), dtype=cfloat
        Returns:
            输出, shape (B, S, D), dtype=cfloat
        """
        B, S, D = x.shape
        H, Dh = self.num_heads, self.d_head

        # ---- 1. 线性投影 + reshape 为多头 ----
        Q = F.linear(x, self.wq).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        K = F.linear(x, self.wk).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        V = F.linear(x, self.wv).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)

        # ---- 2. 复数 RoPE (仅对 Q, K 施加) ----
        Q = self.rotary(Q)  # (B, H, S, Dh) × (1, 1, S, Dh) 复数乘法
        K = self.rotary(K)

        # ---- 3. 共轭干涉: S = Q · K^H / √D_h ----
        # Q^H K: 逐头的 Hermitian 内积矩阵
        # 使用 Q.conj() @ K^T 等价于 (Q^H @ K) 但按 matmul 的维度排列
        scores = torch.matmul(Q, K.transpose(-2, -1).conj())  # (B, H, S, S), ℂ
        scale = 1.0 / math.sqrt(Dh)
        scores = scores * scale

        # ---- 4. MagPhaseSoftmax (幅相分离归一化) ----
        # 模长: |S_{ij}|
        scores_mag = torch.sqrt(
            scores.real ** 2 + scores.imag ** 2 + self.eps ** 2
        )  # (B, H, S, S), ℝ, 数值安全的模长 (避免 |S|=0 处的不连续)

        # 相位因子: S / |S|_ε (单位复数，编码相位关系)
        scores_phase = scores / scores_mag  # (B, H, S, S), ℂ, 单位模长

        # softmax 作用在模长上 (沿 key 维度归一化)
        attn_probs = torch.softmax(scores_mag, dim=-1)  # (B, H, S, S), ℝ

        # 复数注意力权重: 模长归一化 × 相位透传
        # A_{ij} = softmax_j(|S_{ij}|) · e^{i·arg(S_{ij})}
        attn_weights = attn_probs * scores_phase  # (B, H, S, S), ℂ

        # ---- 5. 相干输运: O = A @ V ----
        # Value 被相位旋转后加权叠加，形成全息干涉图样
        attn_out = torch.matmul(attn_weights, V)  # (B, H, S, Dh), ℂ

        # ---- 6. 合并头 + 输出投影 ----
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)
        out = F.linear(attn_out, self.wo)  # (B, S, D), ℂ

        return out


class DCUFFN(nn.Module):
    """
    DCU-FFN (Decoherence Coding Unit 融合的前馈网络)。

    基于平衡零差检测 (balanced homodyne detection) 的 ℂ→ℝ→ℂ 信息处理单元。

    信息流:
        x ∈ ℂ^D                                    (复数输入)
        → W₁_ℂ · x = h ∈ ℂ^M                      (复数模式匹配, M=4D)
        → s = Re(c* ⊙ h) ∈ ℝ^M                    (零差检测: 判据分量)
        → t = Im(c* ⊙ h) / |c|_ε ∈ ℝ^M            (零差检测: 内容分量)
        → o¹ = GELU(s - b) ∈ ℝ^M                   (判据路径: 自门控)
        → o² = σ(s - b) · t ∈ ℝ^M                  (内容路径: 交叉门控)
        → W₂_ℝ · [o¹; o²] ∈ ℝ^{2D}                (实数投影)
        → view_as_complex ∈ ℂ^D                     (重编码为复数)

    零差检测的物理原理:
        信号场 h_m 与本振 c_m = α_m · e^{iθ_m} 在分束器上混合:
            I₊ = |h_m + c_m|², I₋ = |h_m - c_m|²
            差分输出: I₊ - I₋ = 4·Re(c_m* · h_m)
        本振相位 θ_m 选择测量哪个正交分量 (可学习)。
        本振振幅 α_m 控制测量增益 (可学习)。
        差分消除了共模噪声 (|h|² + |c|² 项)。

    双路门控的 QND 测量精神:
        s_m = Re(c_m* h_m): 沿本振方向的投影 → 判据 (测量 X̂)
        t_m = Im(c_m* h_m)/|c_m|: 正交于本振方向 → 内容 (未扰动的 P̂)
        判据路径 GELU(s-b): 自门控，保留幅度信息
        内容路径 σ(s-b)·t: 交叉门控，sigmoid 纯开关
    """

    def __init__(self, d_model: int, ff_mult: int = 4,
                 num_layers: int = 3):
        """
        Args:
            d_model: 复数维度 D
            ff_mult: FFN 扩展倍数 (M = ff_mult × D)
            num_layers: 总层数 L (用于 W₂ 的缩放初始化)
        """
        super().__init__()
        self.d_model = d_model
        self.ff_dim = d_model * ff_mult  # M = 4D
        self.eps = 1e-6

        # ---- Step 1 参数: 复数模式匹配 (线性扩展) ----
        # W₁ ∈ ℂ^{M × D}, 无偏置
        # 每行 w_m 是一个「模式检测器」，h_m = w_m^H x 是输入在该模式上的复数响应
        self.w1 = nn.Parameter(complex_kaiming_init_(
            torch.empty(self.ff_dim, d_model, dtype=torch.cfloat)))

        # ---- Step 2 参数: 本振 (Local Oscillator) ----
        # c ∈ ℂ^M: 每个通道的本振参数
        # |c_m| 编码测量增益, arg(c_m) 编码测量方向
        # 初始化: 单位模长, 均匀随机相位 (均匀覆盖复平面所有方向)
        phases = torch.rand(self.ff_dim) * 2 * math.pi
        c_init = torch.polar(
            torch.ones(self.ff_dim),  # 模长 = 1
            phases                     # 相位 ∈ [0, 2π)
        )
        self.c = nn.Parameter(c_init)

        # ---- Step 3 参数: 门控阈值 ----
        # b ∈ ℝ^M: GELU/sigmoid 的偏移量
        # 初始化为零: 初始时门控完全由 Re(c*h) 的正负性决定
        self.b = nn.Parameter(torch.zeros(self.ff_dim))

        # ---- Step 4 参数: 实数投影 (重编码) ----
        # W₂ ∈ ℝ^{2D × 2M}, 无偏置
        # 将 DCU 的 2M 维实数输出映射到 2D 维实数 (随后 reshape 为 ℂ^D)
        # 缩放初始化: 1/√(2L), 保证训练初期残差修正足够小
        self.w2 = nn.Parameter(scaled_real_kaiming_init_(
            torch.empty(2 * d_model, 2 * self.ff_dim),
            scale=1.0 / math.sqrt(2.0 * num_layers)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 复数模式匹配 → 零差检测 → 双路门控 → 实数投影 → 复数重编码

        Args:
            x: 复数输入 (经 RMSNorm 后), shape (B, S, D), dtype=cfloat
        Returns:
            复数输出, shape (B, S, D), dtype=cfloat
        """
        # ---- Step 1: 复数模式匹配 (线性扩展) ----
        # h = W₁ · x ∈ ℂ^{B×S×M}
        # 每个 h_m = Σ_k W₁[m,k] · x_k 是输入与第 m 个模式检测器的复数内积
        h = F.linear(x, self.w1)  # (B, S, M), ℂ

        # ---- Step 2: 平衡零差检测 ----
        # u = c* ⊙ h (逐元素复数乘法, c 广播到 batch 和 seq 维度)
        # 物理含义: 将信号场 h 旋转到本振 c 的参考系中
        u = self.c.conj() * h  # (B, S, M), ℂ

        # 判据分量: s_m = Re(u_m) = Re(c_m* · h_m)
        # 这是信号沿本振方向的正交投影——平衡零差检测的差分输出
        s = u.real  # (B, S, M), ℝ

        # 内容分量: t_m = Im(u_m) / |c_m|_ε = Im(c_m* · h_m) / |c_m|_ε
        # 这是信号正交于本振方向的分量——QND 测量中未扰动的共轭量
        c_mag = torch.clamp(self.c.abs(), min=self.eps)  # |c_m|_ε, 避免除零
        t = u.imag / c_mag  # (B, S, M), ℝ

        # ---- Step 3: 双路门控 ----
        # 门控输入: s - b (两条路径共享同一组计算结果)
        gate_input = s - self.b  # (B, S, M), ℝ

        # 判据路径 (自门控): o¹ = GELU(s - b)
        # GELU 同时实现门控和信息保留:
        #   s-b >> 0: GELU ≈ s-b (正匹配，保留幅度信息)
        #   s-b << 0: GELU ≈ 0 (反匹配，信号抑制)
        o1 = F.gelu(gate_input)  # (B, S, M), ℝ

        # 内容路径 (交叉门控): o² = σ(s - b) · t
        # sigmoid 纯开关 (不改变 t 的尺度):
        #   s-b >> 0: σ ≈ 1 (门开，内容信号完整通过)
        #   s-b << 0: σ ≈ 0 (门关，内容信号被抑制)
        o2 = torch.sigmoid(gate_input) * t  # (B, S, M), ℝ

        # ---- Step 4: 拼接 + 实数投影 ----
        # 合并两条路径的输出
        o = torch.cat([o1, o2], dim=-1)  # (B, S, 2M), ℝ

        # 实数线性投影: ℝ^{2M} → ℝ^{2D}, 无偏置
        out_real = F.linear(o, self.w2)  # (B, S, 2D), ℝ

        # ---- Step 5: 重整形为复数 ----
        # 将 2D 维实数向量重新解释为 D 维复数向量
        # 前 D 维 → 实部, 后 D 维 → 虚部
        out_complex = torch.view_as_complex(
            out_real.reshape(*out_real.shape[:-1], self.d_model, 2)
        )  # (B, S, D), ℂ

        return out_complex


class ComplexTransformerBlock(nn.Module):
    """
    复数 Transformer Block (Pre-Norm 残差结构)。

    每个 Block 包含两个子层:
    1. HolographicAttention 子层 (纯复数相干操作)
    2. DCU-FFN 子层 (ℂ→ℝ→ℂ 退相干循环)

    每个子层都遵循 Pre-Norm + 残差连接:
        x = x + SubLayer(RMSNorm(x))

    信息流:
        x → Norm → HoloAttn → +x (复数残差)
        → Norm → DCU-FFN → +x (复数残差)
    """

    def __init__(self, d_model: int, num_heads: int = 4,
                 ff_mult: int = 4, num_layers: int = 3):
        """
        Args:
            d_model: 复数维度 D
            num_heads: 注意力头数
            ff_mult: FFN 扩展倍数
            num_layers: 总层数 L (传递给子层用于缩放初始化)
        """
        super().__init__()

        # Attention 子层
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = HolographicAttention(d_model, num_heads, num_layers)

        # DCU-FFN 子层
        self.norm2 = ComplexRMSNorm(d_model)
        self.ffn = DCUFFN(d_model, ff_mult, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 复数输入, shape (B, S, D), dtype=cfloat
        Returns:
            复数输出, shape (B, S, D), dtype=cfloat
        """
        # ---- Attention 子层 ----
        # Pre-Norm: 先归一化再做 Attention
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h  # 复数残差连接

        # ---- DCU-FFN 子层 ----
        # Pre-Norm: 先归一化再做 DCU-FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h  # 复数残差连接

        return x


class HoloDCUByteMLM(nn.Module):
    """
    Holographic-DCU 复数 Transformer — Byte-level MLM。

    完整架构:
        Input → ComplexEmbedding(ℂ^D)
        → [ComplexRMSNorm → HoloAttn(ℂ) → residual
           → ComplexRMSNorm → DCU-FFN(ℂ→ℝ→ℂ) → residual] × L
        → ComplexRMSNorm → Re(x · E^H) + b_cls → logits

    输出层 (Weight Tying):
        logits_v = Re(Σ_k E[v,k]* · x_k) + b_cls[v]
                 = Re(E[v]^H · x) + b_cls[v]

        这等价于: 在展开的实数表示下, 复数 embedding 向量被当作分类权重使用。
        物理含义: 每个 token 的 embedding 作为自己的本振做零差检测——
        logit 衡量的是隐状态与 token embedding 之间的「定向匹配度」。
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 num_heads: int = 4, num_blocks: int = 3,
                 ff_mult: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # ---- Embedding ----
        # vocab_size + 1: 额外一个位置留给 [MASK] token (id = vocab_size)
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)

        # ---- Transformer Blocks ----
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads, ff_mult, num_blocks)
            for _ in range(num_blocks)
        ])

        # ---- 输出层 ----
        # 最终 RMSNorm (在 weight tying 之前归一化)
        self.output_norm = ComplexRMSNorm(d_model)

        # 分类偏置: b_cls ∈ ℝ^V
        # 编码字节频率的先验 (如 Zipf 分布下高频字节的基线 logit)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # 注意: 分类权重复用 self.embedding.weight (weight tying)
        # 不需要额外的 W_cls 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: token ids, shape (B, S), dtype=long
        Returns:
            logits: shape (B, S, V), dtype=float (实数)
        """
        # ---- Embedding ----
        z = self.embedding(x)  # (B, S, D), ℂ

        # ---- L 层 Transformer ----
        for block in self.blocks:
            z = block(z)  # (B, S, D), ℂ

        # ---- 输出: Weight Tying ----
        z = self.output_norm(z)  # (B, S, D), ℂ

        # 取 embedding 表的前 vocab_size 行作为分类权重 (排除 [MASK] token)
        E = self.embedding.weight[:self.vocab_size]  # (V, D), ℂ

        # Hermitian 内积: logits_v = Re(E[v]^H · x)
        # = Re(Σ_k conj(E[v,k]) · z_k)
        # 向量化: z @ conj(E)^T = z @ E^{*T}
        # 然后取实部
        logits = torch.real(
            torch.matmul(z, E.conj().T)  # (B, S, V), ℂ → 取 Re → ℝ
        )

        # 加上分类偏置
        logits = logits + self.output_bias  # (B, S, V), ℝ

        return logits


# =====================================================================
#  实数基线 (与 v7 完全一致)
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
    实数基线 Transformer — Byte-level MLM。

    标准 Pre-Norm Transformer + GELU FFN + weight tying。
    与 v7 的 RealByteMLM 完全一致。
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
#  评估
# =====================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, gen: TextByteGenerator,
                   batch_size: int, device: torch.device,
                   num_batches: int = 10, mask_prob: float = 0.15
                   ) -> Tuple[float, float]:
    """
    统一评估: argmax accuracy + Cross-Entropy perplexity。

    Args:
        model: 待评估模型
        gen: 数据生成器
        batch_size: 批次大小
        device: 计算设备
        num_batches: 评估批次数
        mask_prob: mask 概率

    Returns:
        (accuracy, perplexity) 元组
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
#  统一训练函数
# =====================================================================

def train_model(model: nn.Module, gen: TextByteGenerator,
                cfg: Dict[str, Any], device: torch.device,
                output_dir: str, model_name: str = 'complex'
                ) -> Dict[str, Any]:
    """
    统一训练循环 (autograd, 两种模型共用)。

    Args:
        model: 待训练模型
        gen: 数据生成器
        cfg: 配置字典
        device: 计算设备
        output_dir: 输出目录
        model_name: 模型名称标签

    Returns:
        训练结果字典 (包含 history, 最终指标等)
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr_base = cfg['lr']
    warmup = cfg['warmup_epochs']
    wd = cfg['weight_decay']
    mask_prob = cfg['mask_prob']
    log_interval = cfg['log_interval']

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_base,
        weight_decay=wd, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, max(1, s) / warmup))

    history = {
        'epochs': [], 'loss': [], 'train_acc': [],
        'test_acc': [], 'test_ppl': [], 'lr': [],
    }

    best_train_acc = 0.0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        inp, tgt = gen.generate_train_batch(
            batch_size, mask_mode='bert', mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)
        valid = (tgt != -100)

        logits = model(inp)

        if valid.any():
            loss = F.cross_entropy(logits[valid], tgt[valid])
        else:
            loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % log_interval == 0:
            with torch.no_grad():
                tr_acc = (
                    (logits[valid].argmax(-1) == tgt[valid]).float().mean().item()
                    if valid.any() else 0.0
                )

            test_acc, test_ppl = -1.0, -1.0
            if epoch % (log_interval * 5) == 0 or epoch == epochs - 1:
                test_acc, test_ppl = evaluate_model(
                    model, gen, batch_size, device, num_batches=5)

            history['epochs'].append(epoch)
            history['loss'].append(loss.item())
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
            print(f"  [{model_name}] Ep {epoch:05d} | "
                  f"L: {loss.item():.4f} | Tr: {tr_acc:.2%}"
                  f"{test_str}{ppl_str}")

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
    }
    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  [{model_name}] 完成: Train {best_train_acc:.2%}, "
          f"Test {final_acc:.2%}, PPL {final_ppl:.1f}, {elapsed:.0f}s")
    return result


# =====================================================================
#  主入口
# =====================================================================

DEFAULT_CFG = {
    # 复数模型使用 D=64 (等效 128 实数自由度)
    'd_model_complex': 64,
    # 实数模型使用 d=128 (与 v7 一致)
    'd_model_real': 128,
    'num_heads': 4,
    'num_blocks': 3,
    'ff_mult': 4,
    'seq_len': 64,
    'batch_size': 64,
    'lr': 3e-4,
    'weight_decay': 0.01,
    'epochs': 15000,
    'warmup_epochs': 500,
    'mask_prob': 0.15,
    'log_interval': 200,
}


def main():
    parser = argparse.ArgumentParser(
        description="Byte-level MLM v8: "
                    "HolographicAttention + DCU-FFN 复数 Transformer vs 实数基线")
    parser.add_argument('--model', type=str, default='complex',
                        choices=['complex', 'real', 'both'])
    parser.add_argument('--data-path', type=str, default=None,
                        help="数据文件路径 (默认自动下载 tiny_shakespeare)")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(
                            _ANLA_ROOT, 'Logs', 'nlp_byte_mlm_v8'))
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    cfg = DEFAULT_CFG.copy()
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.seq_len is not None:
        cfg['seq_len'] = args.seq_len
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['lr'] = args.lr

    # 数据加载
    if args.data_path is None:
        data_path = download_tiny_shakespeare(
            os.path.join(_ANLA_ROOT, 'data'))
    else:
        data_path = args.data_path

    gen = TextByteGenerator(
        data_path, cfg['seq_len'], mask_id=256, test_frac=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # ---- 复数模型 (HoloDCU) ----
    if args.model in ('complex', 'both'):
        print("\n" + "=" * 72)
        print("  复数模型 (HolographicAttention + DCU-FFN + Weight Tying)")
        d = cfg['d_model_complex']
        print(f"  d_model={d} (complex), blocks={cfg['num_blocks']}, "
              f"heads={cfg['num_heads']}, ff_mult={cfg['ff_mult']}")
        print("=" * 72)

        model_c = HoloDCUByteMLM(
            vocab_size=256, d_model=d,
            num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
            ff_mult=cfg['ff_mult'],
        ).to(device)

        # 参数量统计
        n_complex = sum(p.numel() for p in model_c.parameters()
                        if p.is_complex())
        n_real = sum(p.numel() for p in model_c.parameters()
                     if not p.is_complex())
        n_total = sum(p.numel() for p in model_c.parameters())
        n_equiv = sum(p.numel() * (2 if p.is_complex() else 1)
                      for p in model_c.parameters())
        print(f"  参数量: 复数 {n_complex:,} + 实数 {n_real:,} "
              f"= 总 {n_total:,}, 实数等效: {n_equiv:,}")

        results['complex'] = train_model(
            model_c, gen, cfg, device,
            os.path.join(args.output_dir, 'complex'), 'complex')

    # ---- 实数基线 ----
    if args.model in ('real', 'both'):
        print("\n" + "=" * 72)
        print("  实数基线 (标准 Transformer — CE + AdamW)")
        d = cfg['d_model_real']
        print(f"  d_model={d} (real), blocks={cfg['num_blocks']}, "
              f"heads={cfg['num_heads']}")
        print("=" * 72)

        model_r = RealByteMLM(
            vocab_size=256, d_model=d,
            num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
            ff_mult=cfg['ff_mult'],
        ).to(device)

        n_params_r = sum(p.numel() for p in model_r.parameters())
        print(f"  参数量: {n_params_r:,}")

        results['real'] = train_model(
            model_r, gen, cfg, device,
            os.path.join(args.output_dir, 'real'), 'real')

    # ---- 对照总结 ----
    if len(results) >= 2:
        print("\n" + "=" * 72)
        print("  对照总结")
        print("=" * 72)
        for name, r in results.items():
            ppl_str = (f", PPL: {r['final_test_ppl']:.1f}"
                       if 'final_test_ppl' in r else "")
            equiv_str = (f", 实数等效: {r['total_real_params']:,}"
                         if 'total_real_params' in r else "")
            print(f"  [{name:>8s}] Test: {r['final_test_acc']:.2%}{ppl_str}"
                  f" | Params: {r['total_params']:,}{equiv_str}"
                  f" | Time: {r['training_time_sec']:.0f}s")

    print(f"\n结果已保存到: {args.output_dir}/")


if __name__ == '__main__':
    main()
