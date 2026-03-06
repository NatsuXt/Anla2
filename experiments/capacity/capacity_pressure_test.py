"""
保存位置: Anla/experiments/capacity/capacity_pressure_test.py

容量压力测试 — 验证 Anla 架构在不同 vocab/d_model 比下的学习能力
=========================================================================

核心思路:
    保持 L_Elegant 不变, 保持 Transformer 架构不变,
    只调整 vocab_size 与 d_model 的比例。

    当 vocab_size >> d_model 时, d_model 维空间放不下那么多正交向量,
    embedding 被迫让拓扑上相邻的 token 共享相似表征。
    这与大模型中 embedding 出现语义簇的机制完全一致。

实验矩阵:
    A: vocab=64,   d=64  (1:1  基线, 已知 100% 无拓扑)
    B: vocab=256,  d=64  (4:1  轻度压力)
    C: vocab=512,  d=32  (16:1 中度压力)
    D: vocab=1024, d=32  (32:1 重度压力)
    E: vocab=1024, d=16  (64:1 极端压力)

评估维度:
    1. 训练准确率 (最近邻匹配)
    2. 泛化准确率 (held-out 起点)
    3. TDA: geodesic H1 与零假设对比
    4. 最近邻环一致率
    5. 相位线性度评分

用法:
    python -m Anla.experiments.capacity.capacity_pressure_test
    python -m Anla.experiments.capacity.capacity_pressure_test --configs A B C
    python -m Anla.experiments.capacity.capacity_pressure_test --configs D --epochs 20000
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

# -------------------------------------------------------------------------
# [Path Fix] 文件位置: Anla/experiments/capacity/capacity_pressure_test.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock


# =====================================================================
#  实验配置矩阵
# =====================================================================
EXPERIMENT_CONFIGS = {
    "A": {
        "name": "A_v64_d64",
        "vocab_size": 64,
        "d_model": 64,
        "num_heads": 4,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 16,
        "lr": 0.001,
        "base_weight_decay": 1e-4,   # [v2] 基准值, 按 d_model 缩放
        "epochs": 5000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,        # [v2] target 侧更新缩放
    },
    "B": {
        "name": "B_v256_d64",
        "vocab_size": 256,
        "d_model": 64,
        "num_heads": 4,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 32,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 10000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
    },
    "C": {
        "name": "C_v512_d32",
        "vocab_size": 512,
        "d_model": 32,
        "num_heads": 4,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 32,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 15000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
    },
    "D": {
        "name": "D_v1024_d32",
        "vocab_size": 1024,
        "d_model": 32,
        "num_heads": 4,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 64,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 20000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
    },
    "E": {
        "name": "E_v1024_d16",
        "vocab_size": 1024,
        "d_model": 16,
        "num_heads": 2,       # d=16 时 4 头太小, 改为 2 头 (每头 8 维)
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 64,
        "lr": 0.001,
        "base_weight_decay": 1e-4,
        "epochs": 20000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
    },
}


# =====================================================================
#  [v2] Weight Decay 缩放
# =====================================================================
def compute_effective_weight_decay(cfg: Dict) -> float:
    """按 d_model 缩放 weight_decay: wd = base_wd * (d_model / 64)"""
    return cfg["base_weight_decay"] * (cfg["d_model"] / 64.0)


# =====================================================================
#  [v2] 平滑 Loss 追踪器
# =====================================================================
class SmoothedLossTracker:
    """基于 EMA 的 Loss 追踪, 用于替代高方差准确率做早停判断。"""
    def __init__(self, ema_beta: float = 0.9):
        self.ema_beta = ema_beta
        self.smoothed_loss = None
        self.best_smoothed_loss = float("inf")
        self.steps_without_improvement = 0

    def update(self, loss: float, interval: int = 1) -> bool:
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = self.ema_beta * self.smoothed_loss + (1.0 - self.ema_beta) * loss
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
#  数据生成器 (支持 held-out 起点)
# =====================================================================
class RingSpanDataGeneratorWithHoldout:
    """
    与原版相同的环形 span masking, 但支持划分 train/test 起点。

    Parameters
    ----------
    vocab_size : int
        环上节点数
    seq_len : int
        序列长度
    mask_id : int
        MASK token ID (= vocab_size)
    holdout_frac : float
        保留为 test 的起点比例 (默认 0.2)
    seed : int
        划分随机种子 (保证可复现)
    """

    def __init__(self, vocab_size: int, seq_len: int, mask_id: int,
                 holdout_frac: float = 0.2, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id

        # 划分 train/test 起点
        rng = random.Random(seed)
        all_starts = list(range(vocab_size))
        rng.shuffle(all_starts)

        n_test = max(1, int(vocab_size * holdout_frac))
        self.test_starts = set(all_starts[:n_test])
        self.train_starts = [s for s in all_starts if s not in self.test_starts]

        # 转为列表以便索引
        self.test_starts_list = sorted(self.test_starts)

    def _generate_batch_from_starts(self, batch_size: int,
                                     starts: list,
                                     max_span: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids_list = []
        target_ids_list = []

        for _ in range(batch_size):
            start = random.choice(starts)
            seq = [(start + i) % self.vocab_size for i in range(self.seq_len)]
            seq_tensor = torch.tensor(seq, dtype=torch.long)

            inp = seq_tensor.clone()
            tgt = torch.full_like(seq_tensor, -100)

            mask_len = random.randint(1, max_span)
            mask_start = random.randint(0, self.seq_len - mask_len)

            inp[mask_start: mask_start + mask_len] = self.mask_id
            tgt[mask_start: mask_start + mask_len] = seq_tensor[mask_start: mask_start + mask_len]

            input_ids_list.append(inp)
            target_ids_list.append(tgt)

        return torch.stack(input_ids_list), torch.stack(target_ids_list)

    def generate_train_batch(self, batch_size: int, max_span: int = 5):
        return self._generate_batch_from_starts(batch_size, self.train_starts, max_span)

    def generate_test_batch(self, batch_size: int, max_span: int = 5):
        return self._generate_batch_from_starts(batch_size, self.test_starts_list, max_span)


# =====================================================================
#  模型 (与 train_ring_masking.py 完全一致)
# =====================================================================
class AnlaManifoldInpainter(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_heads: int):
        super().__init__()
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=128)
        self.block = ComplexTransformerBlock(d_model, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)
        z_out = self.block.forward(z, mask=None)
        return z_out

    def manual_backward(self, force: torch.Tensor, lr: float, wd: float):
        grad = self.block.manual_backward(force, lr, wd)
        grad = self.rotary.manual_backward(grad)
        # [v3] Embedding 层不施加 weight decay (L_Elegant 尺度不变, 无力抵抗收缩)
        self.embedding.manual_backward(grad, lr, weight_decay=0.0)


# =====================================================================
#  L_Elegant (与 train_ring_masking.py 完全一致, 零修改)
# =====================================================================
def compute_elegant_loss_and_force(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple:
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
#  最近邻评估 (支持大 vocab 分块计算, 避免 OOM)
# =====================================================================
@torch.no_grad()
def evaluate_nearest_neighbor(
    z_pred: torch.Tensor,
    target_ids: torch.Tensor,
    all_embeds: torch.Tensor,
    chunk_size: int = 256,
) -> Tuple[float, int, int]:
    valid_mask = (target_ids != -100)
    if not valid_mask.any():
        return 0.0, 0, 0

    z_masked = z_pred[valid_mask]
    true_ids = target_ids[valid_mask]

    vocab_size = all_embeds.shape[0]
    n_masked = z_masked.shape[0]

    # 大 vocab 时分块计算, 避免 (N, V, d) 张量爆显存
    if vocab_size <= 1024:
        dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(dim=-1)
        pred_ids = dists.argmin(dim=-1)
    else:
        pred_ids = torch.empty(n_masked, dtype=torch.long, device=z_masked.device)
        for i in range(0, n_masked, chunk_size):
            end = min(i + chunk_size, n_masked)
            chunk = z_masked[i:end]
            d = (chunk.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(dim=-1)
            pred_ids[i:end] = d.argmin(dim=-1)

    n_correct = (pred_ids == true_ids).sum().item()
    n_total = true_ids.shape[0]
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    return accuracy, n_correct, n_total


# =====================================================================
#  最近邻环一致率
# =====================================================================
@torch.no_grad()
def ring_neighbor_consistency(all_embeds: torch.Tensor, vocab_size: int) -> float:
    """
    对每个 token, 找 embedding 空间中的最近邻,
    检查它是否是环上的 ±1 邻居。
    """
    w = all_embeds[:vocab_size]
    n = w.shape[0]

    # 分块计算距离矩阵的 argmin (排除自身)
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


# =====================================================================
#  相位线性度评分
# =====================================================================
@torch.no_grad()
def phase_linearity_top_scores(all_embeds: torch.Tensor,
                                vocab_size: int,
                                max_freq: int = 4,
                                top_k: int = 5) -> Dict[str, Any]:
    """
    计算每个复数维度与环位置的圆形相关系数。
    返回 top-k 维度的得分及均值。
    """
    w = all_embeds[:vocab_size].cpu()
    phases = torch.angle(w).numpy()
    n, d = phases.shape

    ref_angles = 2.0 * np.pi * np.arange(n) / n

    scores = np.zeros(d, dtype=np.float64)
    for freq in range(1, max_freq + 1):
        ref = np.exp(1j * freq * ref_angles)
        sig = np.exp(1j * phases)
        corr = np.abs(np.mean(sig * ref[:, None].conj(), axis=0))
        scores = np.maximum(scores, corr)

    sorted_idx = np.argsort(-scores)
    top_k_actual = min(top_k, d)

    return {
        "all_scores_mean": float(np.mean(scores)),
        "all_scores_max": float(np.max(scores)),
        "high_linearity_count": int(np.sum(scores > 0.3)),
        "top_dims": [
            {"dim": int(sorted_idx[i]), "score": float(scores[sorted_idx[i]])}
            for i in range(top_k_actual)
        ],
    }


# =====================================================================
#  Embedding 特征导出 (供 validate_tda_loops.py 使用)
# =====================================================================
def export_embedding(model: AnlaManifoldInpainter, vocab_size: int, path: str):
    z = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
    z = np.asarray(z, dtype=np.complex128)
    x_real = np.concatenate([z.real, z.imag], axis=1).astype(np.float64)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, z=z, x_real=x_real)
    return z.shape


# =====================================================================
#  TDA 简易审计 (内置, 不依赖外部 topology_audit_ring.py)
# =====================================================================
def quick_tda_audit(z: np.ndarray, k: int = 6) -> Dict[str, Any]:
    """
    轻量 TDA 审计: 只算 geodesic H1 的基本指标。
    不运行零假设 (那个交给 validate_tda_loops.py 单独做)。
    """
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.csgraph import shortest_path
        from scipy.spatial.distance import pdist, squareform
        from ripser import ripser
    except ImportError:
        return {"available": False, "reason": "ripser/scipy not installed"}

    x = np.concatenate([z.real, z.imag], axis=1).astype(np.float64)
    n = x.shape[0]

    # Euclidean 距离
    De = squareform(pdist(x, metric="euclidean"))

    # kNN geodesic
    k_use = max(1, min(k, n - 1))
    graph = lil_matrix((n, n), dtype=np.float64)
    for i in range(n):
        idx = np.argpartition(De[i], k_use + 1)[:k_use + 1]
        idx = idx[idx != i]
        for j in idx:
            graph[i, j] = De[i, j]
    graph = graph.tocsr()
    graph = graph.maximum(graph.T)
    Dg = shortest_path(graph, directed=False, unweighted=False)
    finite = np.isfinite(Dg)
    if not finite.all():
        max_f = float(np.max(Dg[finite])) if np.any(finite) else 1.0
        Dg[~finite] = max_f * 1.05

    # H1
    result = ripser(Dg, distance_matrix=True, maxdim=1)
    dgm1 = result["dgms"][1]

    if dgm1.size == 0:
        return {
            "available": True,
            "h1_count": 0,
            "dominant_persistence": 0.0,
            "dominance_ratio": 0.0,
        }

    mask = np.isfinite(dgm1[:, 0]) & np.isfinite(dgm1[:, 1])
    dgm1 = dgm1[mask]
    if dgm1.size == 0:
        return {"available": True, "h1_count": 0,
                "dominant_persistence": 0.0, "dominance_ratio": 0.0}

    pers = dgm1[:, 1] - dgm1[:, 0]
    pers = np.sort(pers[pers > 0])[::-1]

    dominant = float(pers[0]) if pers.size > 0 else 0.0
    second = float(pers[1]) if pers.size > 1 else 0.0
    ratio = float(dominant / (second + 1e-12)) if dominant > 0 else 0.0

    return {
        "available": True,
        "h1_count": int(pers.size),
        "dominant_persistence": dominant,
        "second_persistence": second,
        "dominance_ratio": ratio,
    }


# =====================================================================
#  单个配置的训练函数
# =====================================================================
def train_single_config(
    cfg: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    log_interval: int = 200,
    patience: int = 2000,
    eval_test_interval: int = 500,
) -> Dict[str, Any]:
    """
    训练单个配置并返回完整结果。

    Parameters
    ----------
    cfg : dict
        实验配置
    output_dir : str
        输出目录
    device : torch.device
    log_interval : int
        打印间隔
    patience : int
        early stopping 容忍 epoch 数 (准确率无提升则停止)
    eval_test_interval : int
        泛化测试评估间隔

    Returns
    -------
    result : dict
        包含训练/泛化准确率、TDA 指标、时间等
    """
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    seq_len = cfg["seq_len"]
    max_span = cfg["max_span_length"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    mask_id = vocab_size  # MASK token = vocab_size
    reaction_scale = cfg.get("reaction_scale", 0.1)  # [v2]

    # [v2] 按 d_model 缩放 weight_decay
    wd = compute_effective_weight_decay(cfg)

    ratio = vocab_size / d_model

    print()
    print("=" * 72)
    print(f"  Config: {cfg['name']}")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, "
          f"ratio={ratio:.0f}:1, num_heads={num_heads}")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, "
          f"wd={wd:.2e} (scaled), reaction={reaction_scale}")
    print("=" * 72)

    # ---- 数据生成器 ----
    gen = RingSpanDataGeneratorWithHoldout(
        vocab_size=vocab_size,
        seq_len=seq_len,
        mask_id=mask_id,
        holdout_frac=cfg["holdout_frac"],
        seed=42,
    )
    print(f"  Train starts: {len(gen.train_starts)}, "
          f"Test starts (held-out): {len(gen.test_starts_list)}")

    # ---- 模型 ----
    model = AnlaManifoldInpainter(vocab_size, d_model, num_heads).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- 训练记录 ----
    history = {
        "epochs": [],
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "rho": [],           # [v3] 学习进度
        "emb_rms": [],        # [v3] embedding 模长稳定性
    }

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    last_best_epoch = 0

    # [v2] 平滑 Loss 追踪 (用于早停)
    loss_tracker = SmoothedLossTracker(ema_beta=0.9)

    # [v3] ρ(t) 学习进度追踪 (控制 Path B 的有效力度)
    #
    # ρ(t) = clamp((L₀ - L_smoothed(t)) / L₀, 0, 1)
    #   ρ ≈ 0 → 模型未学到东西, prediction 不可靠, Path B 几乎关闭
    #   ρ → 1 → loss 趋近零, prediction 可信, Path B 全力运行
    rho = 0.0
    rho_initial_loss = None     # L₀
    rho_smoothed_loss = None    # L(t), EMA 平滑
    rho_ema_beta = 0.95         # EMA 系数

    t_start = time.time()

    # ---- 训练循环 ----
    for epoch in range(epochs):
        model.train()

        # 生成训练 batch
        input_ids, target_ids = gen.generate_train_batch(batch_size, max_span)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 前向
        z_pred = model.forward(input_ids)

        # 目标原型
        safe_target_ids = target_ids.clone()
        safe_target_ids[target_ids == -100] = 0
        z_target = model.embedding.weight.data[safe_target_ids].detach()

        # L_Elegant
        valid_mask = (target_ids != -100)
        loss_val, force = compute_elegant_loss_and_force(z_pred, z_target, valid_mask)

        # 反向传播
        model.manual_backward(force, lr, wd)

        # [v3] 更新 ρ(t) 学习进度
        if rho_initial_loss is None:
            rho_initial_loss = loss_val
            rho_smoothed_loss = loss_val
        else:
            rho_smoothed_loss = (rho_ema_beta * rho_smoothed_loss
                                 + (1.0 - rho_ema_beta) * loss_val)

        if rho_initial_loss > 1e-12:
            rho = max(0.0, min(1.0,
                    (rho_initial_loss - rho_smoothed_loss) / rho_initial_loss))
        else:
            rho = 0.0

        # [v3] Path B: Target 侧 embedding 更新 (ρ(t) 软缩放)
        # 有效学习率 = lr * reaction_scale * ρ(t)
        reaction_lr = lr * reaction_scale * rho
        if reaction_lr > 1e-12 and valid_mask.any():
            valid_target_ids = target_ids[valid_mask]
            valid_reaction = -force[valid_mask]
            model.embedding.manual_backward_explicit(
                grad=valid_reaction,
                indices=valid_target_ids,
                lr=reaction_lr,
                weight_decay=0.0,  # [v3] embedding 不施加 weight decay
            )

        # ---- 评估 ----
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # 训练集准确率
                z_eval = model.forward(input_ids)
                all_embeds = model.embedding.weight.data[:vocab_size]
                train_acc, n_ok, n_tot = evaluate_nearest_neighbor(
                    z_eval, target_ids, all_embeds)

                # 泛化测试 (每 eval_test_interval 做一次, 或首尾)
                test_acc = -1.0
                if epoch % eval_test_interval == 0 or epoch == epochs - 1:
                    test_input, test_target = gen.generate_test_batch(
                        batch_size, max_span)
                    test_input = test_input.to(device)
                    test_target = test_target.to(device)
                    z_test = model.forward(test_input)
                    test_acc, _, _ = evaluate_nearest_neighbor(
                        z_test, test_target, all_embeds)

            # [v3] 计算 embedding RMS (模长稳定性监控)
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

            # 打印
            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""
            print(f"  [{cfg['name']}] Epoch {epoch:05d} | "
                  f"Loss: {loss_val:.6f} | Train: {train_acc:.2%}{test_str}"
                  f" | ρ: {rho:.3f} | EmbRMS: {emb_rms:.3f}")

            # Early stopping 检查
            if train_acc > best_train_acc or \
               (train_acc == best_train_acc and loss_val < best_loss):
                best_train_acc = train_acc
                best_loss = loss_val
                last_best_epoch = epoch
                epochs_without_improvement = 0

                # 保存最佳 checkpoint
                ckpt_path = os.path.join(output_dir, "best_checkpoint.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "train_acc": train_acc,
                }, ckpt_path)
            else:
                epochs_without_improvement += log_interval

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # [v2] 早停: 基于 smoothed loss 趋势 (替代高方差准确率)
            loss_tracker.update(loss_val, interval=log_interval)

            if loss_tracker.patience_exceeded(patience) and best_train_acc > 0.3:
                print(f"  [Early Stop] Smoothed loss 已 "
                      f"{loss_tracker.steps_without_improvement} epochs "
                      f"无改善, 在 epoch {epoch} 停止 "
                      f"(best acc @ {last_best_epoch})")
                break

            model.train()

    t_elapsed = time.time() - t_start

    # ---- 最终评估 ----
    model.eval()
    with torch.no_grad():
        all_embeds = model.embedding.weight.data[:vocab_size]

        # 最终泛化测试 (多 batch 取均值)
        test_accs = []
        for _ in range(10):
            ti, tt = gen.generate_test_batch(batch_size, max_span)
            ti, tt = ti.to(device), tt.to(device)
            zt = model.forward(ti)
            ta, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds)
            test_accs.append(ta)
        final_test_acc = float(np.mean(test_accs))
        final_test_std = float(np.std(test_accs))

        # 最近邻环一致率
        nn_rate = ring_neighbor_consistency(all_embeds, vocab_size)

        # 相位线性度
        phase_scores = phase_linearity_top_scores(all_embeds, vocab_size)

    # ---- Embedding 导出 + TDA ----
    feat_path = os.path.join(output_dir, "ring_features.npz")
    feat_shape = export_embedding(model, vocab_size, feat_path)

    z_np = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
    z_np = np.asarray(z_np, dtype=np.complex128)
    tda = quick_tda_audit(z_np, k=6)

    # ---- 诊断 ----
    emb_mags = all_embeds.abs().pow(2).mean(dim=1).sqrt().cpu().numpy()

    # ---- 汇总结果 ----
    result = {
        "config": cfg,
        "effective_weight_decay": wd,
        "final_rho": rho,              # [v3] 最终学习进度
        "vocab_d_ratio": ratio,
        "total_params": total_params,
        "training_time_sec": round(t_elapsed, 1),
        "final_epoch": epoch,
        "best_train_acc": best_train_acc,
        "best_train_epoch": last_best_epoch,
        "final_test_acc": final_test_acc,
        "final_test_std": final_test_std,
        "best_test_acc": best_test_acc,
        "ring_nn_consistency": nn_rate,
        "phase_linearity": phase_scores,
        "embedding_mag_mean": float(np.mean(emb_mags)),
        "embedding_mag_std": float(np.std(emb_mags)),
        "tda_quick": tda,
        "features_path": feat_path,
        "history": history,
    }

    # 保存单配置结果
    with open(os.path.join(output_dir, "training_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  [{cfg['name']}] 完成:")
    print(f"    训练准确率: {best_train_acc:.2%} (epoch {last_best_epoch})")
    print(f"    泛化准确率: {final_test_acc:.2%} ± {final_test_std:.2%}")
    print(f"    环邻居一致率: {nn_rate:.1%}")
    print(f"    相位线性度 (max): {phase_scores['all_scores_max']:.4f}")
    print(f"    高线性度维度: {phase_scores['high_linearity_count']}/{d_model}")
    if tda.get("available"):
        print(f"    TDA H1 count: {tda['h1_count']}")
        print(f"    TDA dominant: {tda['dominant_persistence']:.4f}")
        print(f"    TDA ratio: {tda['dominance_ratio']:.4f}")
    print(f"    Weight Decay (effective): {wd:.2e}")
    print(f"    Final ρ: {rho:.3f}")
    print(f"    用时: {t_elapsed:.0f}s")
    print()

    return result


# =====================================================================
#  对比报告生成
# =====================================================================
def generate_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
):
    """生成所有配置的对比摘要。"""

    # ---- JSON 报告 ----
    comparison = {}
    for key, res in results.items():
        cfg = res["config"]
        comparison[key] = {
            "name": cfg["name"],
            "vocab_size": cfg["vocab_size"],
            "d_model": cfg["d_model"],
            "vocab_d_ratio": res["vocab_d_ratio"],
            "best_train_acc": res["best_train_acc"],
            "final_test_acc": res["final_test_acc"],
            "final_test_std": res["final_test_std"],
            "ring_nn_consistency": res["ring_nn_consistency"],
            "phase_linearity_max": res["phase_linearity"]["all_scores_max"],
            "high_linearity_dims": res["phase_linearity"]["high_linearity_count"],
            "tda_h1_count": res["tda_quick"].get("h1_count", -1),
            "tda_dominant": res["tda_quick"].get("dominant_persistence", -1),
            "tda_dominance_ratio": res["tda_quick"].get("dominance_ratio", -1),
            "training_time_sec": res["training_time_sec"],
        }

    report_path = os.path.join(output_dir, "COMPARISON_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # ---- 终端输出 ----
    print()
    print("=" * 90)
    print("  容量压力测试 — 对比报告")
    print("=" * 90)

    header = (f"{'Config':<12} {'V':>5} {'D':>4} {'V/D':>5} "
              f"{'TrainAcc':>9} {'TestAcc':>9} {'NN%':>6} "
              f"{'PhasMax':>8} {'#Hi':>4} "
              f"{'H1cnt':>6} {'H1dom':>8} {'Ratio':>7} {'Time':>6}")
    print(header)
    print("-" * 90)

    for key in sorted(results.keys()):
        c = comparison[key]
        h1_cnt = c["tda_h1_count"] if c["tda_h1_count"] >= 0 else "N/A"
        h1_dom = f"{c['tda_dominant']:.3f}" if c["tda_dominant"] >= 0 else "N/A"
        h1_rat = f"{c['tda_dominance_ratio']:.3f}" if c["tda_dominance_ratio"] >= 0 else "N/A"

        print(f"{c['name']:<12} {c['vocab_size']:>5} {c['d_model']:>4} "
              f"{c['vocab_d_ratio']:>5.0f} "
              f"{c['best_train_acc']:>8.1%} {c['final_test_acc']:>8.1%} "
              f"{c['ring_nn_consistency']:>5.1%} "
              f"{c['phase_linearity_max']:>8.4f} {c['high_linearity_dims']:>4} "
              f"{str(h1_cnt):>6} {h1_dom:>8} {h1_rat:>7} "
              f"{c['training_time_sec']:>5.0f}s")

    print("=" * 90)

    # ---- 自动判断趋势 ----
    keys_sorted = sorted(results.keys(),
                         key=lambda k: results[k]["vocab_d_ratio"])

    nn_rates = [comparison[k]["ring_nn_consistency"] for k in keys_sorted]
    phase_maxes = [comparison[k]["phase_linearity_max"] for k in keys_sorted]

    print()
    print("趋势分析:")

    # 检查环一致率是否随压力增大而提升
    if len(nn_rates) >= 3:
        first_half = np.mean(nn_rates[:len(nn_rates)//2])
        second_half = np.mean(nn_rates[len(nn_rates)//2:])
        if second_half > first_half * 1.5:
            print("  [↑] 环邻居一致率随 vocab/d_model 比增大而显著提升 — "
                  "embedding 在压力下开始编码环拓扑。")
        elif second_half > first_half * 1.1:
            print("  [~] 环邻居一致率有轻微提升趋势, 但不够显著。")
        else:
            print("  [→] 环邻居一致率无明显趋势。")

    if len(phase_maxes) >= 3:
        first_half = np.mean(phase_maxes[:len(phase_maxes)//2])
        second_half = np.mean(phase_maxes[len(phase_maxes)//2:])
        if second_half > first_half * 1.3:
            print("  [↑] 相位线性度随压力增大而提升 — "
                  "相位编码在容量受限时被激活。")
        else:
            print("  [→] 相位线性度无明显趋势。")

    # 泛化 vs 训练差距
    for key in keys_sorted:
        c = comparison[key]
        gap = c["best_train_acc"] - c["final_test_acc"]
        if gap > 0.15 and c["best_train_acc"] > 0.5:
            print(f"  [!] {c['name']}: 训练-泛化差距 {gap:.1%}, 存在过拟合风险。")

    print()
    print(f"完整报告: {report_path}")
    print()

    # ---- 后续命令提示 ----
    print("后续: 对感兴趣的配置运行完整零假设检验:")
    for key in keys_sorted:
        feat_path = results[key]["features_path"]
        name = results[key]["config"]["name"]
        print(f"  # {name}")
        print(f"  python -m Anla.validate_tda_loops \\")
        print(f"      --features {feat_path} --key z \\")
        print(f"      --k 6 --n-null 300 --null-model phase_permute \\")
        print(f"      --out Logs/capacity_pressure_test/config_{name}/LOOP_VALIDATION.json \\")
        print(f"      --plot Logs/capacity_pressure_test/config_{name}/LOOP_VALIDATION.png")
        print()


# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Anla 容量压力测试: 不同 vocab/d_model 比下的学习能力验证"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A", "B", "C", "D", "E"],
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="要运行的配置 (默认: 全部 A B C D E)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖所有配置的 epoch 数 (默认: 使用各配置自身设定)"
    )
    parser.add_argument(
        "--patience", type=int, default=2000,
        help="Early stopping 容忍 epoch 数 (默认: 2000)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=200,
        help="日志打印间隔 (默认: 200)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_ANLA_ROOT, "Logs", "capacity_pressure_test"),
        help="输出根目录"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="计算设备: auto/cpu/cuda"
    )
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 运行
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
            )
            results[config_key] = result
        except Exception as e:
            print(f"\n  [ERROR] Config {config_key} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 对比报告
    if len(results) >= 2:
        generate_comparison_report(results, args.output_dir)
    elif len(results) == 1:
        key = list(results.keys())[0]
        print(f"\n只运行了 1 个配置 ({key}), 跳过对比报告。")
        print(f"结果已保存到: {args.output_dir}/config_{results[key]['config']['name']}/")


if __name__ == "__main__":
    main()
