"""
保存位置: Anla/experiments/capacity/capacity_pressure_test_v4.py

容量压力测试 v4 — Boltzmann-Elegant 损失集成
=========================================================================

v3 → v4 变更:
    [v4] L_Elegant → Boltzmann-Elegant
         · 损失函数从二体力升级为 N 体 Boltzmann 分类
         · Path A: Boltzmann-Elegant 力 (含流形原生排斥力)
         · Path B: 纯 L_Elegant 力 (不变, 仅吸引)
         · 自适应温度 τ (无新超参数)

    动机 (诊断数据):
         · H1: negative_ratio = 42.99% → 近半数预测离错误 token 更近
         · H6: 78.7% 错误在 ring_dist=1 → 邻居混淆, 缺乏排斥力
         · H2/H3 ✅ → Attention/FFN 工作正常, 问题在损失函数

    预期效果:
         · negative_ratio: 43% → < 15%
         · ring_dist_1 错误: 78.7% → < 40%
         · Test Accuracy: 61.9% → 75%+

    所有其他代码 (模型, 数据生成, 评估) 与 v3 完全一致。

用法:
    python -m Anla.experiments.capacity.capacity_pressure_test_v4
    python -m Anla.experiments.capacity.capacity_pressure_test_v4 --configs A
    python -m Anla.experiments.capacity.capacity_pressure_test_v4 --configs A --loss elegant   # 回退到 v3
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
# [Path Fix] 文件位置: Anla/experiments/capacity/capacity_pressure_test_v4.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock

# [v4] Boltzmann-Elegant 导入
from Anla.losses.boltzmann_elegant import compute_boltzmann_elegant_loss_and_force


# =====================================================================
#  实验配置矩阵 (与 v3 完全一致)
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
        "base_weight_decay": 1e-4,
        "epochs": 5000,
        "holdout_frac": 0.2,
        "reaction_scale": 0.1,
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
        "num_heads": 2,
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
#  工具函数 (与 v3 完全一致)
# =====================================================================

def compute_effective_weight_decay(cfg: Dict) -> float:
    return cfg["base_weight_decay"] * (cfg["d_model"] / 64.0)


class SmoothedLossTracker:
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


class RingSpanDataGeneratorWithHoldout:
    def __init__(self, vocab_size, seq_len, mask_id,
                 holdout_frac=0.2, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id

        rng = random.Random(seed)
        all_starts = list(range(vocab_size))
        rng.shuffle(all_starts)

        n_test = max(1, int(vocab_size * holdout_frac))
        self.test_starts = set(all_starts[:n_test])
        self.train_starts = [s for s in all_starts if s not in self.test_starts]
        self.test_starts_list = sorted(self.test_starts)

    def _generate_batch_from_starts(self, batch_size, starts, max_span):
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
            tgt[mask_start: mask_start + mask_len] = \
                seq_tensor[mask_start: mask_start + mask_len]
            input_ids_list.append(inp)
            target_ids_list.append(tgt)
        return torch.stack(input_ids_list), torch.stack(target_ids_list)

    def generate_train_batch(self, batch_size, max_span=5):
        return self._generate_batch_from_starts(
            batch_size, self.train_starts, max_span)

    def generate_test_batch(self, batch_size, max_span=5):
        return self._generate_batch_from_starts(
            batch_size, self.test_starts_list, max_span)


# =====================================================================
#  模型 (与 v3 完全一致)
# =====================================================================
class AnlaManifoldInpainter(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads):
        super().__init__()
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=128)
        self.block = ComplexTransformerBlock(d_model, num_heads=num_heads)

    def forward(self, x):
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)
        z_out = self.block.forward(z, mask=None)
        return z_out

    def manual_backward(self, force, lr, wd):
        grad = self.block.manual_backward(force, lr, wd)
        grad = self.rotary.manual_backward(grad)
        self.embedding.manual_backward(grad, lr, weight_decay=0.0)


# =====================================================================
#  L_Elegant (保留, 用于 --loss elegant 回退模式)
# =====================================================================
def compute_elegant_loss_and_force(z_pred, z_target, valid_mask):
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
#  评估函数 (与 v3 完全一致)
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
    z = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
    z = np.asarray(z, dtype=np.complex128)
    np.savez(path, z=z)
    return z.shape


def quick_tda_audit(z, k=6):
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
#  [v4] 训练循环 — Boltzmann-Elegant 集成
# =====================================================================
def train_single_config(
    cfg: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    log_interval: int = 200,
    patience: int = 2000,
    eval_test_interval: int = 500,
    loss_mode: str = "boltzmann",   # [v4] "boltzmann" or "elegant"
    topk: Optional[int] = None,    # [v4.2] Top-K 聚焦排斥
) -> Dict[str, Any]:
    """
    训练单个配置并返回完整结果。

    Parameters
    ----------
    loss_mode : str
        "boltzmann" — [v4] Boltzmann-Elegant (含排斥力, τ=std(E))
        "elegant"   — [v3] 纯 L_Elegant (仅吸引力, 用于对照)
    topk : int, optional
        Top-K 聚焦排斥。只对最近的 K 个竞争者计算排斥力。
        None = 全 V (默认)。
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
    mask_id = vocab_size
    reaction_scale = cfg.get("reaction_scale", 0.1)

    wd = compute_effective_weight_decay(cfg)
    ratio = vocab_size / d_model

    print()
    print("=" * 72)
    print(f"  Config: {cfg['name']}  |  Loss: {loss_mode.upper()}")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, "
          f"ratio={ratio:.0f}:1, num_heads={num_heads}")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, "
          f"wd={wd:.2e} (scaled), reaction={reaction_scale}")
    print("=" * 72)

    # ---- 数据生成器 ----
    gen = RingSpanDataGeneratorWithHoldout(
        vocab_size=vocab_size, seq_len=seq_len, mask_id=mask_id,
        holdout_frac=cfg["holdout_frac"], seed=42,
    )
    print(f"  Train starts: {len(gen.train_starts)}, "
          f"Test starts (held-out): {len(gen.test_starts_list)}")

    # ---- 模型 ----
    model = AnlaManifoldInpainter(vocab_size, d_model, num_heads).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- [v4.3] τ = std(E_k), 零超参 ----
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
        # [v4] Boltzmann-Elegant 诊断
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

        # 生成训练 batch
        input_ids, target_ids = gen.generate_train_batch(batch_size, max_span)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 前向
        z_pred = model.forward(input_ids)

        # 有效位置掩码
        valid_mask = (target_ids != -100)

        # ============================================================
        #  [v4] 损失计算 — 分支点
        # ============================================================
        if loss_mode == "boltzmann":
            # Boltzmann-Elegant: N 体力 (含排斥)
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

            # Path B: 纯 L_Elegant 力 → target embedding
            # (与 v3 完全一致, 只是从 force_b 取而非重新计算)
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
        #  ρ(t) 更新 (与 v3 完全一致)
        # ============================================================
        # 注: Boltzmann 模式下, loss_val 是 L_BE 而非 L_Elegant
        # 使用 loss_elegant (如果可用) 来计算 ρ, 保持与 v3 的可比性
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
                        batch_size, max_span)
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

            # [v4] Boltzmann 诊断
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

            print(f"  [{cfg['name']}] Epoch {epoch:05d} | "
                  f"Loss: {loss_val:.6f} | Train: {train_acc:.2%}{test_str}"
                  f" | ρ: {rho:.3f} | EmbRMS: {emb_rms:.3f}{extra}")

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
                    "loss_mode": loss_mode,   # [v4] 记录使用的损失类型
                }, ckpt_path)
            else:
                epochs_without_improvement += log_interval

            if test_acc > best_test_acc:
                best_test_acc = test_acc

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
        all_embeds_eval = model.embedding.weight.data[:vocab_size]

        test_accs = []
        for _ in range(10):
            ti, tt = gen.generate_test_batch(batch_size, max_span)
            ti, tt = ti.to(device), tt.to(device)
            zt = model.forward(ti)
            ta, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds_eval)
            test_accs.append(ta)
        final_test_acc = float(np.mean(test_accs))
        final_test_std = float(np.std(test_accs))

        nn_rate = ring_neighbor_consistency(all_embeds_eval, vocab_size)
        phase_scores = phase_linearity_top_scores(all_embeds_eval, vocab_size)

    # ---- Embedding 导出 + TDA ----
    feat_path = os.path.join(output_dir, "ring_features.npz")
    export_embedding(model, vocab_size, feat_path)

    z_np = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
    z_np = np.asarray(z_np, dtype=np.complex128)
    tda = quick_tda_audit(z_np, k=6)

    # ---- 诊断 ----
    emb_mags = all_embeds_eval.abs().pow(2).mean(dim=1).sqrt().cpu().numpy()

    # ---- 汇总结果 ----
    result = {
        "config": cfg,
        "loss_mode": loss_mode,   # [v4]
        "effective_weight_decay": wd,
        "final_rho": rho,
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

    # [v4] 附加 Boltzmann 最终诊断
    # [v4.3] Boltzmann 最终诊断 (τ 来自最终 batch 的 be_info)
    if loss_mode == "boltzmann" and be_info:
        result["final_tau"] = be_info.get("tau", 0)
        result["final_p_target_mean"] = be_info.get("p_target_mean", 0)
        result["final_energy_gap"] = be_info.get("energy_gap_mean", 0)
        result["final_negative_margin_ratio"] = \
            be_info.get("negative_margin_ratio", -1)

    # 保存单配置结果
    with open(os.path.join(output_dir, "training_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  [{cfg['name']}] 完成 (loss={loss_mode}):")
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
    if loss_mode == "boltzmann" and be_info:
        print(f"    Final τ: {be_info.get('tau', 0):.4f}")
    print(f"    用时: {t_elapsed:.0f}s")
    print()

    return result


# =====================================================================
#  对比报告生成 (与 v3 一致, [v4] 追加 Boltzmann 诊断列)
# =====================================================================
def generate_comparison_report(results, output_dir):
    comparison = {}
    for key, res in results.items():
        cfg = res["config"]
        entry = {
            "name": cfg["name"],
            "loss_mode": res.get("loss_mode", "elegant"),
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
        # [v4] Boltzmann 追加
        if "final_tau" in res:
            entry["final_tau"] = res["final_tau"]
            entry["final_p_target_mean"] = res.get("final_p_target_mean", 0)
            entry["final_negative_margin_ratio"] = \
                res.get("final_negative_margin_ratio", -1)
        comparison[key] = entry

    report_path = os.path.join(output_dir, "COMPARISON_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # ---- 终端输出 ----
    print()
    print("=" * 100)
    print("  容量压力测试 v4 — 对比报告")
    print("=" * 100)

    has_boltzmann = any("final_tau" in comparison[k] for k in comparison)

    header = (f"{'Config':<12} {'Loss':<5} {'V':>5} {'D':>4} {'V/D':>5} "
              f"{'TrainAcc':>9} {'TestAcc':>9} {'NN%':>6} "
              f"{'PhasMax':>8} {'#Hi':>4}")
    if has_boltzmann:
        header += f" {'τ':>6} {'p_tgt':>6} {'neg%':>6}"
    header += f" {'Time':>6}"
    print(header)
    print("-" * 100)

    for key in sorted(results.keys()):
        c = comparison[key]
        line = (f"{c['name']:<12} {c['loss_mode'][:5]:<5} "
                f"{c['vocab_size']:>5} {c['d_model']:>4} "
                f"{c['vocab_d_ratio']:>5.0f} "
                f"{c['best_train_acc']:>8.1%} {c['final_test_acc']:>8.1%} "
                f"{c['ring_nn_consistency']:>5.1%} "
                f"{c['phase_linearity_max']:>8.4f} "
                f"{c['high_linearity_dims']:>4}")
        if has_boltzmann:
            tau = c.get("final_tau", 0)
            p_tgt = c.get("final_p_target_mean", 0)
            neg = c.get("final_negative_margin_ratio", -1)
            line += f" {tau:>6.3f} {p_tgt:>6.3f} {neg:>5.1%}"
        line += f" {c['training_time_sec']:>5.0f}s"
        print(line)

    print("=" * 100)
    print(f"\n完整报告: {report_path}\n")


# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Anla 容量压力测试 v4: Boltzmann-Elegant 损失集成"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A"],
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="要运行的配置 (默认: A)"
    )
    parser.add_argument(
        "--loss", type=str, default="boltzmann",
        choices=["boltzmann", "elegant"],
        help="损失函数: boltzmann (v4, 默认) 或 elegant (v3 回退)"
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help="Top-K 聚焦排斥: 只对最近的 K 个竞争者计算排斥力 "
             "(默认: None=全 V)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖所有配置的 epoch 数"
    )
    parser.add_argument(
        "--patience", type=int, default=2000,
        help="Early stopping 容忍 epoch 数"
    )
    parser.add_argument(
        "--log-interval", type=int, default=200,
        help="日志打印间隔"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_ANLA_ROOT, "Logs", "capacity_pressure_test_v4"),
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
                topk=args.topk,
            )
            results[config_key] = result
        except Exception as e:
            print(f"\n  [ERROR] Config {config_key} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) >= 2:
        generate_comparison_report(results, args.output_dir)
    elif len(results) == 1:
        key = list(results.keys())[0]
        print(f"\n只运行了 1 个配置 ({key}), 跳过对比报告。")
        print(f"结果已保存到: "
              f"{args.output_dir}/config_{results[key]['config']['name']}/")

    # 后续命令提示
    print("\n后续:")
    print("  1. 运行诊断探针验证排斥力效果:")
    print("     python -m Anla.diagnostic_probe --config A "
          "--checkpoint-dir Logs/capacity_pressure_test_v4")
    print()
    print("  2. 对比 v3 vs v4 (回退模式):")
    print("     python -m Anla.capacity_pressure_test_v4 "
          "--configs A --loss elegant")
    print()


if __name__ == "__main__":
    main()
