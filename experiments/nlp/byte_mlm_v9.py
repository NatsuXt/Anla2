"""
保存位置: Anla/experiments/nlp/byte_mlm_v9.py

Byte-Level MLM v9 — 相位保真 Transformer (Phase-Faithful ℂ Transformer)
=========================================================================

v9 相对 v8 的两个架构级修改:

    【方案 A】相位保真的 DCU 输出投影
        v8: o¹, o² ∈ ℝ^M → cat → W₂_ℝ ∈ ℝ^{2D×2M} → reshape → ℂ^D
        v9: o¹, o² ∈ ℝ^M → ĥ = o¹ + i·o² ∈ ℂ^M → W₂_ℂ ∈ ℂ^{D×M} → ℂ^D

        根因: v8 的 W₂_ℝ 是自由的实数矩阵，reshape 后不具有 U(1) 等变性。
        复数投影 W₂_ℂ 在实数展开下等价于块结构
            ⎡ A  -B ⎤
            ⎣ B   A ⎦
        自动满足旋转-缩放约束，保证 DCU 输出的相位与输入相位具有
        系统性的代数关系 (而非人工拼接的无结构相位)。

        物理含义: 将 DCU 的"量子反馈"从伪随机相位赋值升级为相干态制备。
        o¹ (判据路径, 幅度信息) 自然对应复数实部,
        o² (内容路径, 正交信息) 自然对应复数虚部——
        与零差检测的正交分量分解在物理上完全自洽。

        参数量变化: W₂ 从 4DM 个实数 → 2DM 个实数 (减半)。
        这不是缺点——是 U(1) 约束剪掉了不应该存在的自由度。

    【方案 B】笛卡尔正交分解注意力 (Cartesian Decomposed Attention)
        v8 MagPhaseSoftmax: A = softmax(|S|) · S/|S|
        v9 CartesianAttn:   A = softmax(Re(S)) · exp(i · Im(S))

        根因: MagPhaseSoftmax 中 S → S/|S| 的归一化导致相位梯度
        ∝ 1/|S|——对被强关注的 key (|S| 大) 相位梯度弱，
        对被弱关注的 key (|S| 小) 相位梯度强但被 softmax 权重截断。
        净效果: 相位学习信号系统性退化。

        v9 的 exp(i·Im(S)) 对 Im(S) 的梯度恒为 i·exp(i·Im(S))，
        模长恒为 1——不依赖 |S|，不退化。

        同时, softmax 输入从 |S| ≥ 0 变为 Re(S) ∈ ℝ:
        负的 Re(S) 表示"强烈不关注"，恢复了标准 attention 的
        注意力稀疏化能力 (v8 MagPhaseSoftmax 中所有 key 至少有
        exp(0)/Σ 的底线概率)。

        物理含义:
            Re(Q^H K) = Re(Q)^T Re(K) + Im(Q)^T Im(K)  → 总体对齐度
            Im(Q^H K) = Im(Q)^T Re(K) - Re(Q)^T Im(K)  → 正交失配
        选择性基于"总体对齐"，旋转角度基于"正交失配"。
        与 DCU 的 Re/Im 分解在概念上一致。

    其余所有组件与 v8 完全一致 (ComplexEmbedding, ComplexRMSNorm,
    ComplexRotaryEmbedding, ComplexTransformerBlock 结构, HoloDCUByteMLM
    输出层, 实数基线, 训练循环, 评估函数)。

用法:
    python -m Anla.experiments.nlp.byte_mlm_v9 --model complex
    python -m Anla.experiments.nlp.byte_mlm_v9 --model real
    python -m Anla.experiments.nlp.byte_mlm_v9 --model both
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
#  数据加载 (与 v8 完全一致)
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
#  工具函数 (与 v8 完全一致)
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


def scaled_complex_kaiming_init_(tensor: torch.Tensor,
                                 scale: float = 1.0) -> torch.Tensor:
    """
    复数 Kaiming 初始化 + 缩放因子。

    用于残差分支的最后一个复数线性层 (W_O, W₂_ℂ)，
    缩放因子 scale = 1/√(2L) 保证训练初期残差修正足够小，
    等价于 GPT-2 / DeepNet 的残差缩放初始化策略。
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
    复数嵌入层。(与 v8 完全一致)

    不做单位圆归一化 (允许模长自由学习)，使用 complex_kaiming_init_ 初始化。
    支持「自发对称性破缺」: 高频 token 可以学到更大的 embedding 模长，
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
    复数 RMSNorm。(与 v8 完全一致)

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
    复数旋转位置编码 (RoPE)。(与 v8 完全一致)

    利用复数乘法的旋转特性，对 Q, K 的每个维度施加位置相关的旋转:
        q_k' = q_k · e^{i · pos · ω_k}

    其中 ω_k = 1/10000^(k/D_h) 是每个维度的旋转频率。
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


# =====================================================================
#  【方案 B】笛卡尔正交分解注意力 (Cartesian Decomposed Attention)
# =====================================================================


class CartesianDecomposedAttention(nn.Module):
    """
    笛卡尔正交分解注意力 — v9 核心创新之一。

    与 v8 HolographicAttention 的区别:

    v8 MagPhaseSoftmax (极坐标分解):
        S = Q^H K / √D_h                                    (复数 score)
        A = softmax(|S|) · S/|S|                             (模长归一化 + 相位透传)

        问题: ∂(S/|S|)/∂S* ∝ 1/|S| → 被强关注的 key 相位梯度弱,
              被弱关注的 key 相位梯度被 softmax 权重截断。净效果: 相位退化。

    v9 CartesianDecomposed (笛卡尔分解):
        S = Q^H K / √D_h                                    (复数 score)
        A = softmax(Re(S)) · exp(i · Im(S))                  (实部选择 + 虚部旋转)

        优势 1: ∂exp(i·Im(S))/∂Im(S) = i·exp(i·Im(S)), |·|=1 恒定, 不退化。
        优势 2: softmax 输入 Re(S) ∈ ℝ 可以为负 → 恢复"强烈不关注"能力
                (v8 的 |S| ≥ 0 让所有 key 至少有 exp(0)/Σ 的底线概率)。

    物理含义:
        Re(Q^H K) = Re(Q)^T Re(K) + Im(Q)^T Im(K)  → 总体对齐度 (选择性)
        Im(Q^H K) = Im(Q)^T Re(K) - Re(Q)^T Im(K)  → 正交失配 (旋转角度)
        注意力基于"你和我有多对齐"来选择，基于"你和我的正交分量差多少"来旋转 Value。
        与 DCU 的 Re/Im 正交分量分离在概念上一致。
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
        # (与 v8 完全一致)
        self.wq = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wk = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wv = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))

        # 输出投影: ℂ^D → ℂ^D, 缩放初始化 1/√(2L)
        # (与 v8 完全一致)
        self.wo = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat),
            scale=1.0 / math.sqrt(2.0 * num_layers)))

        # RoPE (每个头独立的旋转编码)
        # (与 v8 完全一致)
        self.rotary = ComplexRotaryEmbedding(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
            投影 → RoPE → Hermitian 内积 → 笛卡尔正交分解 → 相干输运 → 输出投影

        Args:
            x: 输入, shape (B, S, D), dtype=cfloat
        Returns:
            输出, shape (B, S, D), dtype=cfloat
        """
        B, S, D = x.shape
        H, Dh = self.num_heads, self.d_head

        # ---- 1. 线性投影 + reshape 为多头 ----
        # (与 v8 完全一致)
        Q = F.linear(x, self.wq).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        K = F.linear(x, self.wk).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)
        V = F.linear(x, self.wv).view(B, S, H, Dh).transpose(1, 2)  # (B, H, S, Dh)

        # ---- 2. 复数 RoPE (仅对 Q, K 施加) ----
        # (与 v8 完全一致)
        Q = self.rotary(Q)  # (B, H, S, Dh) × (1, 1, S, Dh) 复数乘法
        K = self.rotary(K)

        # ---- 3. Hermitian 内积: S = Q · K^H / √D_h ----
        # (与 v8 完全一致: 计算复数 score 矩阵)
        scores = torch.matmul(Q, K.transpose(-2, -1).conj())  # (B, H, S, S), ℂ
        scale = 1.0 / math.sqrt(Dh)
        scores = scores * scale

        # ---- 4.【方案 B 修改点】笛卡尔正交分解 ----
        #
        # 取代 v8 的 MagPhaseSoftmax:
        #     v8: A = softmax(|S|) · S/|S|
        #     v9: A = softmax(Re(S)) · exp(i · Im(S))
        #
        # 分解 Hermitian 内积为两个正交分量:
        #     Re(S_{ij}) = (Re(Q)^T Re(K) + Im(Q)^T Im(K)) / √D_h
        #         → "Q 和 K 在所有分量上的总体对齐度"
        #         → 大正值 = 强对齐 → softmax 给高权重
        #         → 大负值 = 强反对齐 → softmax 给极低权重 (v8 做不到!)
        #
        #     Im(S_{ij}) = (Im(Q)^T Re(K) - Re(Q)^T Im(K)) / √D_h
        #         → "Q 和 K 之间的正交失配"
        #         → 编码 Value 被读取时需要施加的相位旋转角度

        # 选择性: softmax 作用在实部上 (沿 key 维度归一化)
        # Re(S) ∈ ℝ 可以为负——恢复标准 attention 的"强烈不关注"能力
        attn_probs = torch.softmax(scores.real, dim=-1)  # (B, H, S, S), ℝ

        # 相位旋转: exp(i · Im(S))
        # 梯度: ∂exp(iθ)/∂θ = i·exp(iθ), 模长恒为 1, 不退化
        phase_rotors = torch.polar(
            torch.ones_like(scores.imag),  # 模长 = 1
            scores.imag                     # 旋转角 = Im(S)
        )  # (B, H, S, S), ℂ, 单位模长

        # 复数注意力权重: 实部选择 × 虚部旋转
        # A_{ij} = softmax_j(Re(S_{ij})) · exp(i · Im(S_{ij}))
        attn_weights = attn_probs * phase_rotors  # (B, H, S, S), ℂ

        # ---- 5. 相干输运: O = A @ V ----
        # Value 被相位旋转后加权叠加
        # (与 v8 完全一致)
        attn_out = torch.matmul(attn_weights, V)  # (B, H, S, Dh), ℂ

        # ---- 6. 合并头 + 输出投影 ----
        # (与 v8 完全一致)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)
        out = F.linear(attn_out, self.wo)  # (B, S, D), ℂ

        return out


# =====================================================================
#  【方案 A】相位保真 DCU-FFN (Phase-Faithful DCU-FFN)
# =====================================================================


class PhaseFaithfulDCUFFN(nn.Module):
    """
    相位保真 DCU-FFN — v9 核心创新之二。

    与 v8 DCUFFN 的区别仅在 Step 4-5 (输出投影):

    v8 信息流:
        x ∈ ℂ^D → W₁_ℂ → h ∈ ℂ^M → DCU → o¹, o² ∈ ℝ^M
        → cat([o¹; o²]) ∈ ℝ^{2M} → W₂_ℝ ∈ ℝ^{2D×2M} → ℝ^{2D} → reshape → ℂ^D

    v9 信息流:
        x ∈ ℂ^D → W₁_ℂ → h ∈ ℂ^M → DCU → o¹, o² ∈ ℝ^M
        → ĥ = o¹ + i·o² ∈ ℂ^M → W₂_ℂ ∈ ℂ^{D×M} → ℂ^D

    物理含义的精确对应:
        o¹_m = GELU(s_m - b_m):
            判据路径输出——沿本振方向的投影强度 (幅度信息)
            → 自然对应复数的实部
        o²_m = σ(s_m - b_m) · t_m:
            内容路径输出——正交于本振方向的信号分量 (相位信息)
            → 自然对应复数的虚部
        ĥ_m = o¹_m + i · o²_m:
            零差检测的完整测量结果 (复数形式)
            |ĥ_m| = √(o¹² + o²²) 是测量总信号强度
            arg(ĥ_m) = arctan(o²/o¹) 是判据-内容比值角

    W₂_ℂ 的 U(1) 等变性:
        复数矩阵乘法 W₂_ℂ · ĥ 在实数展开下等价于:
            ⎡ A  -B ⎤ ⎡ o¹ ⎤   ⎡ Ao¹ - Bo² ⎤
            ⎣ B   A ⎦ ⎣ o² ⎦ = ⎣ Bo¹ + Ao² ⎦
        这保证了输出的相位与输入的相位具有系统性的代数关系——
        对 ĥ 施加全局相位旋转 e^{iφ}，输出也精确旋转 e^{iφ}。
        v8 的 W₂_ℝ 不具有此性质。
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
        # W₁ ∈ ℂ^{M × D}, 无偏置
        # (与 v8 完全一致)
        self.w1 = nn.Parameter(complex_kaiming_init_(
            torch.empty(self.ff_dim, d_model, dtype=torch.cfloat)))

        # ---- Step 2 参数: 本振 (Local Oscillator) ----
        # c ∈ ℂ^M: 每个通道的本振参数
        # (与 v8 完全一致: 单位模长, 均匀随机相位)
        phases = torch.rand(self.ff_dim) * 2 * math.pi
        c_init = torch.polar(
            torch.ones(self.ff_dim),  # 模长 = 1
            phases                     # 相位 ∈ [0, 2π)
        )
        self.c = nn.Parameter(c_init)

        # ---- Step 3 参数: 门控阈值 ----
        # b ∈ ℝ^M: GELU/sigmoid 的偏移量
        # (与 v8 完全一致: 初始化为零)
        self.b = nn.Parameter(torch.zeros(self.ff_dim))

        # ---- Step 4 参数:【方案 A 修改点】复数投影 (相干态制备) ----
        # v8: W₂ ∈ ℝ^{2D × 2M}   (实数, 自由度 4DM)
        # v9: W₂ ∈ ℂ^{D × M}     (复数, 自由度 2DM)
        #
        # 将 DCU 的测量结果 ĥ ∈ ℂ^M 通过复数线性映射投影回 ℂ^D。
        # 复数矩阵自动满足 U(1) 等变性——输出相位与输入相位有系统性关系。
        #
        # 缩放初始化: 1/√(2L), 与 v8 的 W₂_ℝ 的缩放策略一致,
        # 保证训练初期残差修正足够小。
        self.w2 = nn.Parameter(scaled_complex_kaiming_init_(
            torch.empty(d_model, self.ff_dim, dtype=torch.cfloat),
            scale=1.0 / math.sqrt(2.0 * num_layers)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
            复数模式匹配 → 零差检测 → 双路门控 → 复数重组 → 复数投影

        Args:
            x: 复数输入 (经 RMSNorm 后), shape (B, S, D), dtype=cfloat
        Returns:
            复数输出, shape (B, S, D), dtype=cfloat
        """
        # ---- Step 1: 复数模式匹配 (线性扩展) ----
        # h = W₁ · x ∈ ℂ^{B×S×M}
        # (与 v8 完全一致)
        h = F.linear(x, self.w1)  # (B, S, M), ℂ

        # ---- Step 2: 平衡零差检测 ----
        # (与 v8 完全一致)
        # u = c* ⊙ h: 将信号场 h 旋转到本振 c 的参考系中
        u = self.c.conj() * h  # (B, S, M), ℂ

        # 判据分量: s_m = Re(c_m* · h_m)
        # 信号沿本振方向的正交投影——平衡零差检测的差分输出
        s = u.real  # (B, S, M), ℝ

        # 内容分量: t_m = Im(c_m* · h_m) / |c_m|_ε
        # 信号正交于本振方向的分量——QND 测量中未扰动的共轭量
        c_mag = torch.clamp(self.c.abs(), min=self.eps)  # |c_m|_ε
        t = u.imag / c_mag  # (B, S, M), ℝ

        # ---- Step 3: 双路门控 ----
        # (与 v8 完全一致)
        gate_input = s - self.b  # (B, S, M), ℝ

        # 判据路径 (自门控): o¹ = GELU(s - b)
        o1 = F.gelu(gate_input)  # (B, S, M), ℝ

        # 内容路径 (交叉门控): o² = σ(s - b) · t
        o2 = torch.sigmoid(gate_input) * t  # (B, S, M), ℝ

        # ---- Step 4:【方案 A 修改点】复数重组 + 复数投影 ----
        #
        # v8 此处是:
        #     o = cat([o1, o2], dim=-1)        # ℝ^{2M}
        #     out_real = F.linear(o, self.w2)  # W₂_ℝ: ℝ^{2M} → ℝ^{2D}
        #     out = view_as_complex(reshape)    # → ℂ^D
        #
        # v9 改为:
        #     ĥ = o1 + i·o2                    # 复数重组: ℝ^M × ℝ^M → ℂ^M
        #     out = W₂_ℂ · ĥ                   # 复数投影: ℂ^M → ℂ^D

        # 复数重组: 将判据路径 (实部) 和内容路径 (虚部) 合成为复数测量结果
        # 物理含义: o1 是幅度信号 → Re, o2 是正交信号 → Im
        # |ĥ_m| = √(o1² + o2²) 是第 m 个通道的测量总信号强度
        # arg(ĥ_m) = arctan(o2_m / o1_m) 是判据-内容比值角
        h_measured = torch.complex(o1, o2)  # (B, S, M), ℂ

        # 复数投影: ℂ^M → ℂ^D (相干态制备)
        # W₂_ℂ 是复数矩阵，保证了 U(1) 等变性:
        #   对 ĥ 施加 e^{iφ} 旋转 → 输出也精确旋转 e^{iφ}
        out = F.linear(h_measured, self.w2)  # (B, S, D), ℂ

        return out


# =====================================================================
#  复数 Transformer Block (使用 v9 组件)
# =====================================================================


class ComplexTransformerBlock(nn.Module):
    """
    复数 Transformer Block (Pre-Norm 残差结构)。

    结构与 v8 完全一致，但使用 v9 的两个新组件:
    1. CartesianDecomposedAttention 替代 HolographicAttention  【方案 B】
    2. PhaseFaithfulDCUFFN 替代 DCUFFN                        【方案 A】

    每个子层都遵循 Pre-Norm + 残差连接:
        x = x + SubLayer(RMSNorm(x))

    信息流:
        x → Norm → CartesianAttn → +x (复数残差)
        → Norm → PhaseFaithful-DCU-FFN → +x (复数残差)
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

        # Attention 子层:【方案 B】CartesianDecomposedAttention
        self.norm1 = ComplexRMSNorm(d_model)
        self.attn = CartesianDecomposedAttention(d_model, num_heads, num_layers)

        # FFN 子层:【方案 A】PhaseFaithfulDCUFFN
        self.norm2 = ComplexRMSNorm(d_model)
        self.ffn = PhaseFaithfulDCUFFN(d_model, ff_mult, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 复数输入, shape (B, S, D), dtype=cfloat
        Returns:
            复数输出, shape (B, S, D), dtype=cfloat
        """
        # ---- Attention 子层 ----
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h  # 复数残差连接

        # ---- FFN 子层 ----
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h  # 复数残差连接

        return x


# =====================================================================
#  完整模型: HoloDCU v9 Byte-level MLM
# =====================================================================


class HoloDCUByteMLM(nn.Module):
    """
    Phase-Faithful 复数 Transformer — Byte-level MLM。

    完整架构:
        Input → ComplexEmbedding(ℂ^D)
        → [ComplexRMSNorm → CartesianAttn(ℂ) → residual
           → ComplexRMSNorm → PF-DCU-FFN(ℂ→ℝ→ℂ) → residual] × L
        → ComplexRMSNorm → Re(x · E^H) + b_cls → logits

    输出层 (Weight Tying) 与 v8 完全一致:
        logits_v = Re(Σ_k E[v,k]* · x_k) + b_cls[v]
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

        # ---- Transformer Blocks (使用 v9 组件) ----
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads, ff_mult, num_blocks)
            for _ in range(num_blocks)
        ])

        # ---- 输出层 (与 v8 完全一致) ----
        self.output_norm = ComplexRMSNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

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

        # 取 embedding 表的前 vocab_size 行作为分类权重
        E = self.embedding.weight[:self.vocab_size]  # (V, D), ℂ

        # Hermitian 内积: logits_v = Re(E[v]^H · x)
        logits = torch.real(
            torch.matmul(z, E.conj().T)  # (B, S, V)
        )

        # 加上分类偏置
        logits = logits + self.output_bias  # (B, S, V), ℝ

        return logits


# =====================================================================
#  实数基线 (与 v8 完全一致, 不做任何修改)
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
    与 v7/v8 的 RealByteMLM 完全一致。
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
#  评估 (与 v8 完全一致)
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
#  统一训练函数 (与 v8 完全一致)
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
    # (与 v8 完全一致)
    'd_model_complex': 64,
    # 实数模型使用 d=128 (与 v7/v8 一致)
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
        description="Byte-level MLM v9: "
                    "CartesianAttn + PhaseFaithful-DCU-FFN 复数 Transformer "
                    "vs 实数基线")
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
                            _ANLA_ROOT, 'Logs', 'nlp_byte_mlm_v9'))
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

    # ---- 复数模型 (v9: CartesianAttn + PhaseFaithful-DCU-FFN) ----
    if args.model in ('complex', 'both'):
        print("\n" + "=" * 72)
        print("  复数模型 v9 (CartesianDecomposedAttn + PhaseFaithful-DCU-FFN)")
        d = cfg['d_model_complex']
        print(f"  d_model={d} (complex), blocks={cfg['num_blocks']}, "
              f"heads={cfg['num_heads']}, ff_mult={cfg['ff_mult']}")
        print("=" * 72)

        model_c = HoloDCUByteMLM(
            vocab_size=256, d_model=d,
            num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
            ff_mult=cfg['ff_mult'],
        ).to(device)

        # 参数量统计 (v9 因方案 A 的 W₂_ℂ 参数量减少)
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

    # ---- 实数基线 (与 v8 完全一致) ----
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
