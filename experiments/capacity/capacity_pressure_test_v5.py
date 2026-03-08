"""
保存位置: Anla/experiments/capacity/capacity_pressure_test_v5.py

容量压力测试 v5 — Encoder-HoloTransformer-Decoder 架构
=========================================================================

v4 → v5 架构变更:

    [1] Encoder-HoloTransformer-Decoder 架构
        旧 (v4): Embedding → Rotary → TransformerBlock(×1) → z_pred
        新 (v5): Embedding → Rotary → Encoder → Block(×3) → Decoder → z_pred

        设计思想 (参考 VAE):
            Encoder: Embedding 空间 → Transformer 潜在空间 (ComplexLinear + RMSNorm)
            HoloTransformerBlocks: 在潜在空间中进行推理 (×3)
            Decoder: 潜在空间 → Embedding 空间 (ComplexLinear, 含 bias)

        核心价值:
            a) 解耦 Embedding 空间与 Transformer 工作空间
               — Embedding 专注于 token 身份编码
               — Transformer 有自己的"思考空间", 不受 Embedding 几何约束
            b) Decoder 的权重尺度隐式控制 E_k 的绝对大小
               — 部分消解 τ 的设计难题 (等效于 logit 投影头)
            c) Decoder bias 提供复平面平移
               — 对齐 Transformer 输出分布与 Embedding 分布的中心

    [2] BERT 风格多位置 Masking
        旧: 每个序列 mask 一个连续 span (长度 1~5)
        新: 每个位置以概率 p_mask=0.15 独立 mask

        信息密度提升:
            旧: 每 batch 约 96 个有效训练位置 (batch=32, 平均 span=3)
            新: 每 batch 约 154 个有效训练位置 (batch=32, seq=32, 0.15)
            提升约 60%, 每步有更多 token 获得梯度信号。

        通用性:
            不针对任何特定拓扑结构, 是标准 MLM 训练策略。

    [3] PhaseTwist 初始化 γ=0.1, β=0.1
        (由 activation.py v5 实现, 此处不需额外代码)

    [4] Path A 径向投影 (project_out_radial)
        (由 embedding.py v5 + base_layer.py v5 实现, 此处不需额外代码)

    [5] 线性 Warmup 学习率调度
        lr(t) = lr_base · min(1, t / T_warmup)
        T_warmup = 500 epochs
        给 Polar Adam 的二阶矩估计一个热身期。

    [6] Block 数量: 1 → 3
        单层 Attention + FFN 只能做一阶推理。
        3 层允许: 定位上下文 → 整合信息 → 输出准备。

用法:
    python -m Anla.experiments.capacity.capacity_pressure_test_v5
    python -m Anla.experiments.capacity.capacity_pressure_test_v5 --configs B
    python -m Anla.experiments.capacity.capacity_pressure_test_v5 --configs B --mask-mode span  # 回退到 v4 span mask
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# -------------------------------------------------------------------------
# [Path Fix]
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.linear import ComplexLinear
from Anla.layers.normalization import ComplexRMSNorm
from Anla.losses.boltzmann_elegant import compute_boltzmann_elegant_loss_and_force


# =====================================================================
#  实验配置矩阵
# =====================================================================
# [v5] 与 v4 一致, 但 epochs 适当增大 (3 Block 需要更多训练步)
EXPERIMENT_CONFIGS = {
    "A": {
        "name": "A_v64_d64",
        "vocab_size": 64,
        "d_model": 64,
        "num_heads": 4,
        "num_blocks": 3,           # [v5] 1 → 3
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 16,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 8000,            # [v5] 5000 → 8000
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
        "warmup_epochs": 500,      # [v5] 新增
        "mask_prob": 0.15,         # [v5] 新增
    },
    "B": {
        "name": "B_v256_d64",
        "vocab_size": 256,
        "d_model": 64,
        "num_heads": 4,
        "num_blocks": 3,           # [v5] 1 → 3
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 32,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 15000,           # [v5] 10000 → 15000
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
        "warmup_epochs": 500,      # [v5] 新增
        "mask_prob": 0.15,         # [v5] 新增
    },
    "C": {
        "name": "C_v512_d32",
        "vocab_size": 512,
        "d_model": 32,
        "num_heads": 4,
        "num_blocks": 3,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 32,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 20000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
        "warmup_epochs": 500,
        "mask_prob": 0.15,
    },
    "D": {
        "name": "D_v1024_d32",
        "vocab_size": 1024,
        "d_model": 32,
        "num_heads": 4,
        "num_blocks": 3,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 64,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 25000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
        "warmup_epochs": 500,
        "mask_prob": 0.15,
    },
}


# =====================================================================
#  工具函数
# =====================================================================

def compute_effective_weight_decay(cfg: Dict) -> float:
    """与 v4 一致: weight_decay 随 d_model 线性缩放。"""
    return cfg["base_weight_decay"] * (cfg["d_model"] / 64.0)


def compute_warmup_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    """
    [v5] 线性 Warmup 学习率调度。

    lr(t) = lr_base · min(1, t / T_warmup)

    动机:
        训练初期 Polar Adam 的二阶矩 (v_r, v_t) 缺乏足够的统计信息,
        使用全量学习率会导致更新步长过大。
        逐渐增大学习率给优化器一个"热身"期。
        这在实数网络中是标准做法 (BERT, GPT 等均使用 warmup)。

    Args:
        base_lr:       目标学习率
        epoch:         当前 epoch
        warmup_epochs: warmup 周期

    Returns:
        当前 epoch 的有效学习率
    """
    if epoch >= warmup_epochs:
        return base_lr
    # 线性 warmup: 从 base_lr/warmup_epochs 线性增长到 base_lr
    # 保证 epoch=0 时 lr > 0 (取 max(1, epoch))
    return base_lr * max(1, epoch) / warmup_epochs


class SmoothedLossTracker:
    """与 v4 完全一致。"""
    def __init__(self, ema_beta: float = 0.9):
        self.ema_beta = ema_beta
        self.smoothed_loss = None
        self.best_smoothed_loss = float("inf")
        self.steps_without_improvement = 0

    def update(self, loss: float, interval: int = 1) -> bool:
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = (self.ema_beta * self.smoothed_loss
                                  + (1.0 - self.ema_beta) * loss)
        improved = False
        if self.smoothed_loss < self.best_smoothed_loss - 1e-8:
            self.best_smoothed_loss = self.smoothed_loss
            self.steps_without_improvement = 0
            improved = True
        else:
            self.steps_without_improvement += interval
        return improved

    def patience_exceeded(self, patience: int) -> bool:
        return self.steps_without_improvement >= patience


# =====================================================================
#  [v5] 数据生成器 — 支持 BERT 风格多位置 mask
# =====================================================================
class RingDataGenerator:
    """
    Ring 数据生成器, 支持两种 mask 模式:

    [1] 'bert' — BERT 风格多位置独立 mask (v5 默认)
        每个位置以概率 p_mask 独立决定是否 mask。
        优势: 更高的信息密度, 更多 token 获得训练信号。

    [2] 'span' — 连续 span mask (v4 兼容)
        随机选择一个位置, mask 连续 1~max_span 个 token。
        用于与 v4 对照。

    两种模式的数据生成 (ring 序列) 完全相同,
    只有 mask 策略不同。
    """

    def __init__(self, vocab_size: int, seq_len: int, mask_id: int,
                 holdout_frac: float = 0.2, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id

        # Train/Test 分割 (与 v4 完全一致)
        rng = random.Random(seed)
        all_starts = list(range(vocab_size))
        rng.shuffle(all_starts)

        n_test = max(1, int(vocab_size * holdout_frac))
        self.test_starts = set(all_starts[:n_test])
        self.train_starts = [s for s in all_starts if s not in self.test_starts]
        self.test_starts_list = sorted(self.test_starts)

    def _generate_batch(self, batch_size: int, starts: list,
                        mask_mode: str = 'bert',
                        mask_prob: float = 0.15,
                        max_span: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成一个 batch 的 ring 序列。

        Args:
            batch_size: batch 大小
            starts:     可用的起始 token 列表
            mask_mode:  'bert' 或 'span'
            mask_prob:  [bert 模式] 每个位置的 mask 概率
            max_span:   [span 模式] 最大 span 长度

        Returns:
            input_ids:  (B, S) — mask 位置被替换为 mask_id
            target_ids: (B, S) — 非 mask 位置为 -100
        """
        input_ids_list = []
        target_ids_list = []

        for _ in range(batch_size):
            # 生成 ring 序列
            start = random.choice(starts)
            seq = [(start + i) % self.vocab_size for i in range(self.seq_len)]
            seq_tensor = torch.tensor(seq, dtype=torch.long)

            inp = seq_tensor.clone()
            tgt = torch.full_like(seq_tensor, -100)

            if mask_mode == 'bert':
                # ----- BERT 风格: 每个位置独立 mask -----
                # 以概率 mask_prob 选择要 mask 的位置
                mask_flags = torch.rand(self.seq_len) < mask_prob
                # 保证至少有 1 个 mask 位置 (避免空 batch)
                if not mask_flags.any():
                    mask_flags[random.randint(0, self.seq_len - 1)] = True
                # 应用 mask
                inp[mask_flags] = self.mask_id
                tgt[mask_flags] = seq_tensor[mask_flags]

            elif mask_mode == 'span':
                # ----- Span 风格: 连续 span mask (v4 兼容) -----
                mask_len = random.randint(1, max_span)
                mask_start = random.randint(0, self.seq_len - mask_len)
                inp[mask_start: mask_start + mask_len] = self.mask_id
                tgt[mask_start: mask_start + mask_len] = \
                    seq_tensor[mask_start: mask_start + mask_len]

            else:
                raise ValueError(f"Unknown mask_mode: {mask_mode}")

            input_ids_list.append(inp)
            target_ids_list.append(tgt)

        return torch.stack(input_ids_list), torch.stack(target_ids_list)

    def generate_train_batch(self, batch_size: int,
                             mask_mode: str = 'bert',
                             mask_prob: float = 0.15,
                             max_span: int = 5):
        return self._generate_batch(batch_size, self.train_starts,
                                    mask_mode, mask_prob, max_span)

    def generate_test_batch(self, batch_size: int,
                            mask_mode: str = 'bert',
                            mask_prob: float = 0.15,
                            max_span: int = 5):
        return self._generate_batch(batch_size, self.test_starts_list,
                                    mask_mode, mask_prob, max_span)


# =====================================================================
#  [v5] 模型 — Encoder-HoloTransformer-Decoder
# =====================================================================
class AnlaManifoldInpainter_v5(nn.Module):
    """
    v5 Encoder-HoloTransformer-Decoder 架构。

    信号路径:
        input_ids
         → ComplexEmbedding(V+1, D)    — token → 复数流形嵌入
         → ComplexRotaryEmbedding(D)   — 旋转位置编码
         → Encoder:
             ComplexLinear(D, D)        — Embedding 空间 → 潜在空间
             ComplexRMSNorm(D)          — 归一化, 为第一个 Block 提供标准尺度
         → TransformerBlock × num_blocks — 潜在空间中的多层推理
         → Decoder:
             ComplexLinear(D, D, bias=True) — 潜在空间 → Embedding 空间
         → z_pred

    Encoder 的设计 (全纯映射, 保角变换):
        Encoder 不包含激活函数, 作为全纯函数实现 Embedding 空间
        到潜在空间的保角变换。ComplexLinear 是复数矩阵乘法,
        在复数分析中是全纯映射 (Holomorphic mapping),
        保角性质使得 Embedding 空间中的角度关系在潜在空间中得以保留。
        非线性由后续 Block 内的 PhaseTwist 提供。
        未来可探索: 添加激活函数引入非全纯性的效果。

    Decoder 的设计:
        纯线性层 + bias。不加 RMSNorm (否则输出模长被归一化,
        Decoder 无法通过模长传递信息)。不加激活函数 (非线性会
        扭曲流形结构, 让最近邻解码不可靠)。
        bias 提供复平面平移, 对齐 Transformer 输出分布中心
        与 Embedding 分布中心。

    反向传播路径:
        Path A (模型权重更新):
            force_a → decoder.backward → block[-1].backward → ...
            → block[0].backward → encoder_norm.backward
            → encoder_linear.backward → rotary.backward
            → embedding.backward(project_out_radial=True)

        Path B (Embedding 直接更新):
            force_b → embedding.manual_backward_explicit()
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_blocks: int = 3):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_blocks = num_blocks

        # ---- Embedding + 位置编码 ----
        # vocab_size + 1: 额外的 MASK token
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=128)

        # ---- Encoder: Embedding 空间 → 潜在空间 ----
        # ComplexLinear: 全纯映射 (保角变换)
        # ComplexRMSNorm: 归一化输出, 为第一个 Block 提供标准尺度
        self.encoder_linear = ComplexLinear(d_model, d_model, bias=False)
        self.encoder_norm = ComplexRMSNorm(d_model)

        # ---- HoloTransformer Blocks ----
        # [v5] num_blocks 个 TransformerBlock, 各自独立
        # PhaseTwist 使用 v5 默认初始化 (γ=0.1, β=0.1)
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads=num_heads, ff_mult=4)
            for _ in range(num_blocks)
        ])

        # ---- Decoder: 潜在空间 → Embedding 空间 ----
        # 纯线性 + bias, 无 Norm, 无激活
        # bias=True: 提供复平面平移, 对齐分布中心
        self.decoder = ComplexLinear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: input_ids, (B, S), long tensor

        Returns:
            z_pred: (B, S, D), complex tensor — 预测的流形坐标
        """
        # 1. Embedding + 位置编码
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)

        # 2. Encoder: Embedding 空间 → 潜在空间
        z = self.encoder_linear.forward(z)
        z = self.encoder_norm.forward(z)

        # 3. HoloTransformer: 多层推理
        for block in self.blocks:
            z = block.forward(z, mask=None)

        # 4. Decoder: 潜在空间 → Embedding 空间
        z_pred = self.decoder.forward(z)

        return z_pred

    def manual_backward(self, force: torch.Tensor, lr: float, wd: float):
        """
        Path A: 完整反向传播 — 更新所有模型权重。

        梯度路径 (逆序):
            force → decoder → blocks[-1] → ... → blocks[0]
            → encoder_norm → encoder_linear → rotary → embedding

        [v5] Embedding 层使用 project_out_radial=True:
            移除 Path A 梯度的径向分量, 仅更新相位。
            模长由 Path B (Boltzmann 力场) 控制。
        """
        # Decoder 反向
        grad = self.decoder.manual_backward(force, lr, 0.0)

        # Blocks 反向 (逆序)
        for block in reversed(self.blocks):
            grad = block.manual_backward(grad, lr, wd)

        # Encoder 反向
        grad = self.encoder_norm.manual_backward(grad, lr)
        grad = self.encoder_linear.manual_backward(grad, lr, 0.0)

        # Rotary 反向
        grad = self.rotary.manual_backward(grad)

        # Embedding 反向
        # [v5] project_out_radial=True — 由 embedding.manual_backward 内部实现
        self.embedding.manual_backward(grad, lr, weight_decay=0.0)


# =====================================================================
#  L_Elegant (保留, 用于 --loss elegant 回退模式)
# =====================================================================
def compute_elegant_loss_and_force(z_pred, z_target, valid_mask):
    """与 v4 完全一致。"""
    eps = 1e-8
    r = z_pred.abs() + eps
    r_hat = z_target.abs() + eps
    u = z_pred / r
    u_hat = z_target / r_hat

    log_ratio = torch.log(r) - torch.log(r_hat)
    loss_mag = log_ratio.pow(2)
    loss_phase = (u - u_hat).abs().pow(2)
    loss_elem = loss_mag + loss_phase

    force_radial = (log_ratio / r) * u
    force_tangential = (u * u * u_hat.conj() - u_hat) / (2.0 * r)
    force = force_radial + force_tangential

    mask_3d = valid_mask.unsqueeze(-1).to(force.dtype)
    force = force * mask_3d
    loss_elem = loss_elem * mask_3d.real

    num_valid = valid_mask.sum().float().clamp(min=1.0)
    force = force / num_valid
    loss_scalar = loss_elem.sum() / (num_valid * z_pred.shape[-1])

    return loss_scalar.item(), force


# =====================================================================
#  评估函数 (与 v4 完全一致)
# =====================================================================
@torch.no_grad()
def evaluate_nearest_neighbor(z_pred, target_ids, all_embeds, chunk_size=256):
    valid_mask = (target_ids != -100)
    if not valid_mask.any():
        return 0.0, 0, 0

    z_masked = z_pred[valid_mask]
    true_ids = target_ids[valid_mask]
    vocab_size = all_embeds.shape[0]
    n_masked = z_masked.shape[0]

    if vocab_size <= 1024:
        dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)) \
            .abs().pow(2).sum(dim=-1)
        pred_ids = dists.argmin(dim=-1)
    else:
        pred_ids = torch.empty(n_masked, dtype=torch.long,
                               device=z_masked.device)
        for i in range(0, n_masked, chunk_size):
            end = min(i + chunk_size, n_masked)
            chunk = z_masked[i:end]
            d = (chunk.unsqueeze(1) - all_embeds.unsqueeze(0)) \
                .abs().pow(2).sum(dim=-1)
            pred_ids[i:end] = d.argmin(dim=-1)

    n_correct = (pred_ids == true_ids).sum().item()
    n_total = true_ids.shape[0]
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return accuracy, n_correct, n_total


@torch.no_grad()
def ring_neighbor_consistency(all_embeds, vocab_size):
    """与 v4 完全一致。"""
    w = all_embeds[:vocab_size]
    n = w.shape[0]
    nn_ids = torch.empty(n, dtype=torch.long, device=w.device)
    for i in range(n):
        dists = (w[i].unsqueeze(0) - w).abs().pow(2).sum(dim=-1)
        dists[i] = float("inf")
        nn_ids[i] = dists.argmin()
    nn_ids = nn_ids.cpu().numpy()
    token_ids = np.arange(n)
    prev_n = (token_ids - 1) % n
    next_n = (token_ids + 1) % n
    is_ring = (nn_ids == prev_n) | (nn_ids == next_n)
    return float(is_ring.mean())


@torch.no_grad()
def phase_linearity_top_scores(all_embeds, vocab_size, max_freq=4, top_k=5):
    """与 v4 完全一致。"""
    w = all_embeds[:vocab_size]
    n, d = w.shape
    phases = torch.angle(w)
    phase_diffs = phases[1:] - phases[:-1]
    phase_diffs = torch.atan2(torch.sin(phase_diffs), torch.cos(phase_diffs))

    per_dim_scores = []
    for dim_idx in range(d):
        diffs = phase_diffs[:, dim_idx]
        std_val = diffs.std().item()
        score = max(0.0, 1.0 - std_val / 1.5)
        per_dim_scores.append(score)

    per_dim_scores_t = torch.tensor(per_dim_scores)
    topk_vals, topk_idx = per_dim_scores_t.topk(min(top_k, d))

    return {
        "top_dims": topk_idx.tolist(),
        "top_scores": topk_vals.tolist(),
        "all_scores_max": max(per_dim_scores),
        "all_scores_mean": float(np.mean(per_dim_scores)),
        "high_linearity_count": sum(1 for s in per_dim_scores if s > 0.5),
    }


def export_embedding(model, vocab_size, path):
    """与 v4 完全一致。"""
    z = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
    z = np.asarray(z, dtype=np.complex128)
    np.savez(path, z=z)
    return z.shape


def quick_tda_audit(z, k=6):
    """与 v4 完全一致。"""
    try:
        from Anla.analysis.validate_tda_loops import compute_geodesic_persistence
        result = compute_geodesic_persistence(z, k=k)
        h1 = result["h1_pairs"]
        if len(h1) == 0:
            return {"available": True, "h1_count": 0,
                    "dominant_persistence": 0, "dominance_ratio": 0}
        persistences = sorted([d - b for b, d in h1], reverse=True)
        return {
            "available": True,
            "h1_count": len(h1),
            "dominant_persistence": persistences[0],
            "dominance_ratio": (persistences[0] / persistences[1]
                                if len(persistences) > 1 else float("inf")),
        }
    except Exception:
        return {"available": False}


# =====================================================================
#  [v5] 训练循环
# =====================================================================
def train_single_config(
    cfg: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    log_interval: int = 200,
    patience: int = 3000,          # [v5] 2000 → 3000 (3 Block 收敛更慢)
    eval_test_interval: int = 500,
    loss_mode: str = "boltzmann",
    mask_mode: str = "bert",       # [v5] 新增: 'bert' 或 'span'
    topk: Optional[int] = None,
) -> Dict[str, Any]:
    """
    训练单个配置并返回完整结果。

    Parameters
    ----------
    loss_mode : str
        "boltzmann" — Boltzmann-Elegant (含排斥力)
        "elegant"   — 纯 L_Elegant (用于对照)
    mask_mode : str
        "bert"  — [v5] BERT 风格多位置独立 mask
        "span"  — [v4] 连续 span mask
    topk : int, optional
        Top-K 聚焦排斥。None = 全 V (默认)。
    """
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    num_blocks = cfg.get("num_blocks", 3)
    seq_len = cfg["seq_len"]
    max_span = cfg["max_span_length"]
    batch_size = cfg["batch_size"]
    lr_base = cfg["lr"]
    epochs = cfg["epochs"]
    mask_id = vocab_size
    reaction_scale = cfg.get("reaction_scale", 0.1)
    warmup_epochs = cfg.get("warmup_epochs", 500)
    mask_prob = cfg.get("mask_prob", 0.15)

    wd = compute_effective_weight_decay(cfg)
    ratio = vocab_size / d_model

    print()
    print("=" * 72)
    print(f"  Config: {cfg['name']}  |  Loss: {loss_mode.upper()}"
          f"  |  Mask: {mask_mode.upper()}")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, "
          f"ratio={ratio:.0f}:1, num_heads={num_heads}")
    print(f"  num_blocks={num_blocks}, warmup={warmup_epochs}, "
          f"mask_prob={mask_prob:.0%}")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr_base}, "
          f"wd={wd:.2e} (scaled), reaction={reaction_scale}")
    print(f"  [v5] Encoder-Decoder 架构, PhaseTwist γ=0.1 β=0.1, "
          f"Path A radial projection")
    print("=" * 72)

    # ---- 数据生成器 ----
    gen = RingDataGenerator(
        vocab_size=vocab_size, seq_len=seq_len, mask_id=mask_id,
        holdout_frac=cfg["holdout_frac"], seed=42,
    )
    print(f"  Train starts: {len(gen.train_starts)}, "
          f"Test starts (held-out): {len(gen.test_starts_list)}")

    # ---- 模型 ----
    model = AnlaManifoldInpainter_v5(
        vocab_size, d_model, num_heads, num_blocks
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    if loss_mode == "boltzmann":
        topk_str = f"topk={topk}" if topk is not None else "topk=all"
        print(f"  τ: std(E_k) — 零超参 ({topk_str})")

    # ---- 训练记录 ----
    history = {
        "epochs": [],
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "rho": [],
        "emb_rms": [],
        "lr": [],                   # [v5] 记录实际 lr
        # Boltzmann-Elegant 诊断
        "tau": [],
        "p_target_mean": [],
        "energy_gap": [],
        "negative_margin_ratio": [],
        "loss_elegant": [],
    }

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    last_best_epoch = 0

    loss_tracker = SmoothedLossTracker(ema_beta=0.9)

    # ρ(t) 学习进度追踪
    rho = 0.0
    rho_initial_loss = None
    rho_smoothed_loss = None
    rho_ema_beta = 0.95

    t_start = time.time()

    # ---- 训练循环 ----
    for epoch in range(epochs):
        model.train()

        # [v5] Warmup 学习率
        lr = compute_warmup_lr(lr_base, epoch, warmup_epochs)

        # 生成训练 batch
        input_ids, target_ids = gen.generate_train_batch(
            batch_size, mask_mode=mask_mode,
            mask_prob=mask_prob, max_span=max_span,
        )
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 前向
        z_pred = model.forward(input_ids)

        # 有效位置掩码
        valid_mask = (target_ids != -100)

        # ============================================================
        #  损失计算
        # ============================================================
        if loss_mode == "boltzmann":
            all_embeds = model.embedding.weight.data[:vocab_size].detach()

            loss_val, force_a, force_b, be_info = \
                compute_boltzmann_elegant_loss_and_force(
                    z_pred=z_pred,
                    all_embeds=all_embeds,
                    target_ids=target_ids,
                    valid_mask=valid_mask,
                    topk=topk,
                )

            # Path A: Boltzmann-Elegant 力 → 模型权重
            model.manual_backward(force_a, lr, wd)

            # Path B: Boltzmann 力 → target embedding
            reaction_lr = lr * reaction_scale * rho
            if reaction_lr > 1e-12 and valid_mask.any():
                valid_target_ids = target_ids[valid_mask]
                valid_reaction = -force_b[valid_mask]
                model.embedding.manual_backward_explicit(
                    grad=valid_reaction,
                    indices=valid_target_ids,
                    lr=reaction_lr,
                    weight_decay=0.0,
                )

        else:
            # v3 回退: 纯 L_Elegant (二体力)
            safe_target_ids = target_ids.clone()
            safe_target_ids[target_ids == -100] = 0
            z_target = model.embedding.weight.data[safe_target_ids].detach()

            loss_val, force = compute_elegant_loss_and_force(
                z_pred, z_target, valid_mask)

            model.manual_backward(force, lr, wd)

            reaction_lr = lr * reaction_scale * rho
            if reaction_lr > 1e-12 and valid_mask.any():
                valid_target_ids = target_ids[valid_mask]
                valid_reaction = -force[valid_mask]
                model.embedding.manual_backward_explicit(
                    grad=valid_reaction,
                    indices=valid_target_ids,
                    lr=reaction_lr,
                    weight_decay=0.0,
                )

            be_info = {}

        # ============================================================
        #  ρ(t) 更新 (与 v4 完全一致)
        # ============================================================
        rho_loss = be_info.get("loss_elegant", loss_val)

        if rho_initial_loss is None:
            rho_initial_loss = rho_loss
            rho_smoothed_loss = rho_loss
        else:
            rho_smoothed_loss = (rho_ema_beta * rho_smoothed_loss
                                 + (1.0 - rho_ema_beta) * rho_loss)

        if rho_initial_loss > 1e-12:
            rho = max(0.0, min(1.0,
                    (rho_initial_loss - rho_smoothed_loss) / rho_initial_loss))
        else:
            rho = 0.0

        # ---- 评估 ----
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                all_embeds_eval = model.embedding.weight.data[:vocab_size]

                train_acc, n_ok, n_tot = evaluate_nearest_neighbor(
                    z_pred, target_ids, all_embeds_eval)

                test_acc = -1.0
                if epoch % eval_test_interval == 0 or epoch == epochs - 1:
                    test_input, test_target = gen.generate_test_batch(
                        batch_size, mask_mode=mask_mode,
                        mask_prob=mask_prob, max_span=max_span,
                    )
                    test_input = test_input.to(device)
                    test_target = test_target.to(device)
                    z_test = model.forward(test_input)
                    test_acc, _, _ = evaluate_nearest_neighbor(
                        z_test, test_target, all_embeds_eval)

            emb_rms = torch.sqrt(
                model.embedding.weight.data[:vocab_size].abs().pow(2).mean()
            ).item()

            # 记录
            history["epochs"].append(epoch)
            history["loss"].append(loss_val)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["rho"].append(rho)
            history["emb_rms"].append(emb_rms)
            history["lr"].append(lr)

            # Boltzmann 诊断
            history["tau"].append(be_info.get("tau", 0))
            history["p_target_mean"].append(be_info.get("p_target_mean", 0))
            history["energy_gap"].append(be_info.get("energy_gap_mean", 0))
            history["negative_margin_ratio"].append(
                be_info.get("negative_margin_ratio", -1))
            history["loss_elegant"].append(be_info.get("loss_elegant", loss_val))

            # 打印
            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""

            if loss_mode == "boltzmann":
                tau_str = f" | τ: {be_info['tau']:.3f}"
                p_str = f" | p_tgt: {be_info['p_target_mean']:.3f}"
                neg_str = f" | neg%: {be_info['negative_margin_ratio']:.1%}"
                extra = f"{tau_str}{p_str}{neg_str}"
            else:
                extra = ""

            lr_str = f" | lr: {lr:.5f}" if epoch < warmup_epochs else ""

            print(f"  [{cfg['name']}] Epoch {epoch:05d} | "
                  f"Loss: {loss_val:.6f} | Train: {train_acc:.2%}{test_str}"
                  f" | ρ: {rho:.3f} | EmbRMS: {emb_rms:.3f}{extra}{lr_str}")

            # Early stopping 检查
            if train_acc > best_train_acc or \
               (train_acc == best_train_acc and loss_val < best_loss):
                best_train_acc = train_acc
                best_loss = loss_val
                last_best_epoch = epoch
                epochs_without_improvement = 0

                ckpt_path = os.path.join(output_dir, "best_checkpoint.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "loss_mode": loss_mode,
                    "mask_mode": mask_mode,    # [v5]
                    "version": "v5",           # [v5]
                }, ckpt_path)
            else:
                epochs_without_improvement += log_interval

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Early stopping
            loss_tracker.update(loss_val, interval=log_interval)
            # [v4.4] 双重判据: acc AND loss 同时停滞才触发
            if epochs_without_improvement >= patience and \
               loss_tracker.patience_exceeded(patience):
                print(f"  [{cfg['name']}] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    t_elapsed = time.time() - t_start

    # ================================================================
    #  导出与分析
    # ================================================================

    # Best checkpoint 分析
    best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pth")
    best_metrics = None
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # 导出 best embedding
        shape = export_embedding(model, vocab_size,
                                 os.path.join(output_dir, "ring_features_best.npz"))

        # 评估
        model.eval()
        with torch.no_grad():
            all_embeds_eval = model.embedding.weight.data[:vocab_size]
            nn_rate = ring_neighbor_consistency(all_embeds_eval, vocab_size)
            phase_scores = phase_linearity_top_scores(all_embeds_eval, vocab_size)

            # 多次测试平均
            test_accs = []
            for _ in range(5):
                ti, tt = gen.generate_test_batch(
                    batch_size, mask_mode=mask_mode,
                    mask_prob=mask_prob, max_span=max_span,
                )
                ti, tt = ti.to(device), tt.to(device)
                zt = model.forward(ti)
                ta, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds_eval)
                test_accs.append(ta)

        z_np = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
        z_np = np.asarray(z_np, dtype=np.complex128)
        tda = quick_tda_audit(z_np)

        best_metrics = {
            "test_acc": float(np.mean(test_accs)),
            "test_std": float(np.std(test_accs)),
            "nn_rate": nn_rate,
            "phase_scores": phase_scores,
            "tda": tda,
        }

    # Final model 分析
    # (重新加载最后状态 — 如果做了 early stopping, 模型已经是 best)
    model.eval()
    with torch.no_grad():
        all_embeds_eval = model.embedding.weight.data[:vocab_size]
        final_nn_rate = ring_neighbor_consistency(all_embeds_eval, vocab_size)
        final_phase = phase_linearity_top_scores(all_embeds_eval, vocab_size)

        test_accs_final = []
        for _ in range(5):
            ti, tt = gen.generate_test_batch(
                batch_size, mask_mode=mask_mode,
                mask_prob=mask_prob, max_span=max_span,
            )
            ti, tt = ti.to(device), tt.to(device)
            zt = model.forward(ti)
            ta, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds_eval)
            test_accs_final.append(ta)

    export_embedding(model, vocab_size,
                     os.path.join(output_dir, "ring_features_final.npz"))

    final_metrics = {
        "test_acc": float(np.mean(test_accs_final)),
        "test_std": float(np.std(test_accs_final)),
        "nn_rate": final_nn_rate,
        "phase_scores": final_phase,
    }

    # 使用 best_metrics 如果可用, 否则用 final
    primary = best_metrics if best_metrics else {
        **final_metrics,
        "tda": quick_tda_audit(
            model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
                .astype(np.complex128)
        ),
    }

    # ================================================================
    #  结果打包
    # ================================================================
    result = {
        "config": cfg,
        "version": "v5",            # [v5]
        "loss_mode": loss_mode,
        "mask_mode": mask_mode,      # [v5]
        "vocab_d_ratio": ratio,
        "best_train_acc": best_train_acc,
        "last_best_epoch": last_best_epoch,
        "total_epochs": epoch + 1,
        "final_test_acc": primary["test_acc"],
        "final_test_std": primary["test_std"],
        "ring_nn_consistency": primary["nn_rate"],
        "phase_linearity": primary["phase_scores"],
        "tda_quick": primary["tda"],
        "training_time_sec": t_elapsed,
        "history": history,
    }

    # Boltzmann 诊断
    if loss_mode == "boltzmann" and be_info:
        result["final_tau"] = be_info.get("tau", 0)
        result["final_p_target_mean"] = be_info.get("p_target_mean", 0)
        result["final_energy_gap"] = be_info.get("energy_gap_mean", 0)
        result["final_negative_margin_ratio"] = \
            be_info.get("negative_margin_ratio", -1)

    # 保存
    with open(os.path.join(output_dir, "training_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  [{cfg['name']}] 完成 (loss={loss_mode}, mask={mask_mode}):")
    print(f"    训练准确率: {best_train_acc:.2%} (epoch {last_best_epoch})")
    print(f"    泛化准确率: {primary['test_acc']:.2%} ± {primary['test_std']:.2%}")
    print(f"    环邻居一致率: {primary['nn_rate']:.1%}")
    print(f"    相位线性度 (max): {primary['phase_scores']['all_scores_max']:.4f}")
    print(f"    总参数量: {total_params:,}")
    print(f"    用时: {t_elapsed:.0f}s")
    if loss_mode == "boltzmann" and be_info:
        print(f"    Final τ: {be_info.get('tau', 0):.4f}")
    print()

    return result


# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Anla 容量压力测试 v5: Encoder-Decoder 架构"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A"],
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="要运行的配置 (默认: A)"
    )
    parser.add_argument(
        "--loss", type=str, default="boltzmann",
        choices=["boltzmann", "elegant"],
        help="损失函数: boltzmann (默认) 或 elegant (回退)"
    )
    parser.add_argument(
        "--mask-mode", type=str, default="bert",
        choices=["bert", "span"],
        help="[v5] mask 模式: bert (默认, 多位置独立) 或 span (v4 兼容)"
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="Top-K 聚焦排斥 (默认: None=全 V)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖所有配置的 epoch 数"
    )
    parser.add_argument(
        "--patience", type=int, default=3000,
        help="Early stopping 容忍 epoch 数"
    )
    parser.add_argument(
        "--log-interval", type=int, default=200,
        help="日志打印间隔"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_ANLA_ROOT, "Logs", "capacity_pressure_test_v5"),
        help="输出根目录"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="计算设备: auto/cpu/cuda"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Loss mode: {args.loss}")
    print(f"Mask mode: {args.mask_mode}")
    if args.topk is not None:
        print(f"Top-K: {args.topk}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for config_key in args.configs:
        cfg = EXPERIMENT_CONFIGS[config_key].copy()
        if args.epochs is not None:
            cfg["epochs"] = args.epochs

        config_dir = os.path.join(args.output_dir, f"config_{cfg['name']}")

        try:
            result = train_single_config(
                cfg=cfg,
                output_dir=config_dir,
                device=device,
                log_interval=args.log_interval,
                patience=args.patience,
                loss_mode=args.loss,
                mask_mode=args.mask_mode,
                topk=args.topk,
            )
            results[config_key] = result
        except Exception as e:
            print(f"\n  [ERROR] Config {config_key} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) >= 1:
        key = list(results.keys())[0]
        print(f"\n结果已保存到: "
              f"{args.output_dir}/config_{results[key]['config']['name']}/")

    print("\n后续:")
    print("  1. 可视化: python -m Anla.visualization.visualize_config_B "
          "<output_dir>")
    print("  2. 对比 v4: python -m Anla.experiments.capacity."
          "capacity_pressure_test_v5 --mask-mode span")
    print()


if __name__ == "__main__":
    main()
