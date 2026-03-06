"""
保存位置: Anla/losses/boltzmann_elegant.py

Boltzmann-Elegant 损失函数 (v4.3)
=========================================================================

核心公式:

    E_k(z) = Σ_d [ ln²(r_d / r̂_{k,d}) + |u_d - û_{k,d}|² ]

    τ_n = std_k(E_k^{(n)})     — per-sample 温度, 零超参

    p_k = softmax(-E_k / τ)

    L = -log p_target

温度 τ = std(E) 的推导:
    softmax 的区分力取决于 logit 差的标准差:
        std(logit) = std(E) / τ
    要求 std(logit) = O(1) → τ = std(E)
    等价于对 softmax 输入做标准化 (与 LayerNorm 同理)。

完整 Wirtinger 梯度 (含 τ-throughput):

    dL/dz* = (1/τ)(F_target - Σ p_k F_k)
           + [(Ē_p - E_target) / (V · τ³)] · Σ_k (E_k - Ē) F_k

    其中:
        F_k = ∂E_k/∂z*        — L_Elegant 流形力
        Ē_p = Σ p_k E_k       — Boltzmann 加权平均能量
        Ē   = (1/V) Σ E_k     — 算术平均能量
        V    = 词表大小

    第一项: 标准 Boltzmann-Elegant 力 (吸引 - 排斥)
    第二项: τ 对 z 的依赖产生的修正力 (dτ/dz* 穿透项)

    推导第二项:
        ∂L/∂τ = (Ē_p - E_target) / τ²
        dτ/dz* = d(std(E))/dz* = [1/(V·τ)] Σ_k (E_k - Ē) F_k
        第二项 = (∂L/∂τ)(dτ/dz*)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


def compute_boltzmann_elegant_loss_and_force(
    z_pred: torch.Tensor,
    all_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: Optional[int] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Boltzmann-Elegant 损失。零超参, 完整 Wirtinger 梯度。

    Parameters
    ----------
    z_pred : (B, S, D), complex
    all_embeds : (V, D), complex — detached embedding 快照
    target_ids : (B, S), long — 非 MASK 位置为 -100
    valid_mask : (B, S), bool
    topk : int, optional — Top-K 聚焦排斥

    Returns
    -------
    loss_val, force_a, force_b, info
    """
    eps = 1e-8
    B, S, D = z_pred.shape
    V = all_embeds.shape[0]

    # ================================================================
    #  Step 0: 提取有效位置
    # ================================================================

    z_valid = z_pred[valid_mask]              # (N, D)
    true_ids = target_ids[valid_mask]         # (N,)
    N = z_valid.shape[0]

    if N == 0:
        zero_force = torch.zeros_like(z_pred)
        return 0.0, zero_force, zero_force.clone(), {
            "tau": 0, "p_target_mean": 0, "effective_k": 0,
        }

    # ================================================================
    #  Step 1: E_k — 流形能量 (N, V)
    # ================================================================

    z_exp = z_valid.unsqueeze(1)              # (N, 1, D)
    e_exp = all_embeds.unsqueeze(0)           # (1, V, D)

    r = z_exp.abs() + eps
    r_hat = e_exp.abs() + eps
    u = z_exp / r
    u_hat = e_exp / r_hat

    log_ratio = torch.log(r) - torch.log(r_hat)
    energy_mag = log_ratio.pow(2)
    energy_phase = (u - u_hat).abs().pow(2)

    E_k = (energy_mag + energy_phase).sum(dim=-1)  # (N, V)

    # ================================================================
    #  Step 2: F_k — 流形力 (N, V, D)
    # ================================================================

    F_radial = (log_ratio / r) * u
    F_tangential = (u * u * u_hat.conj() - u_hat) / (2.0 * r)
    F_k = F_radial + F_tangential                  # (N, V, D)

    # ================================================================
    #  Step 3: Target 量
    # ================================================================

    idx_n = torch.arange(N, device=z_pred.device)
    E_target = E_k[idx_n, true_ids]                # (N,)
    F_target = F_k[idx_n, true_ids]                # (N, D)

    # ================================================================
    #  Step 3.5: Top-K 聚焦排斥
    # ================================================================

    use_topk = (topk is not None) and (topk < V)
    effective_k = topk if use_topk else V

    if use_topk:
        _, topk_idx = E_k.topk(topk, dim=-1, largest=False)
        topk_mask = torch.zeros(N, V, dtype=torch.bool, device=E_k.device)
        topk_mask.scatter_(1, topk_idx, True)
        topk_mask[idx_n, true_ids] = True
    else:
        topk_mask = None

    # ================================================================
    #  Step 4: τ_n = std_k(E_k^{(n)}) — per-sample 温度
    # ================================================================

    if topk_mask is not None:
        E_masked = E_k.masked_fill(~topk_mask, float('nan'))
        E_bar = torch.nanmean(E_masked, dim=-1, keepdim=True)     # (N, 1)
        tau_per = torch.nanmean(
            (E_masked - E_bar).pow(2), dim=-1
        ).sqrt() + eps                                              # (N,)
        V_eff = topk_mask.sum(dim=-1).float()                      # (N,)
    else:
        E_bar = E_k.mean(dim=-1, keepdim=True)                     # (N, 1)
        tau_per = E_k.std(dim=-1) + eps                             # (N,)
        V_eff = torch.full((N,), float(V), device=E_k.device)      # (N,)

    tau_col = tau_per.unsqueeze(-1)                 # (N, 1) for broadcasting

    # ================================================================
    #  Step 5: Boltzmann 概率与损失
    # ================================================================

    logits = -E_k / tau_col                         # (N, V)

    if topk_mask is not None:
        logits = logits.masked_fill(~topk_mask, float('-inf'))

    loss_per_sample = F.cross_entropy(logits, true_ids, reduction='none')
    loss_val = loss_per_sample.mean().item()

    p_k = torch.softmax(logits, dim=-1)             # (N, V)

    # ================================================================
    #  Step 6: 完整 Wirtinger 梯度 (含 τ-throughput)
    # ================================================================
    #
    # Term 1: 标准 Boltzmann-Elegant 力 (τ 视为常数)
    #   (1/τ)(F_target - Σ p_k F_k)
    #
    # Term 2: τ-throughput 修正
    #   (∂L/∂τ)(dτ/dz*)
    #   = [(Ē_p - E_target) / (V_eff · τ³)] · Σ_k (E_k - Ē) F_k

    # --- Term 1 ---
    weighted_F = (p_k.unsqueeze(-1) * F_k).sum(dim=1)   # (N, D)
    term1 = (1.0 / tau_col) * (F_target - weighted_F)    # (N, D)

    # --- Term 2: τ-throughput ---
    # ∂L/∂τ = (Ē_p - E_target) / τ²
    E_p_mean = (p_k * E_k).sum(dim=-1)                   # (N,)
    dL_dtau = (E_p_mean - E_target) / (tau_per ** 2)      # (N,)

    # dτ/dz* = [1/(V_eff · τ)] Σ_k (E_k - Ē) F_k
    E_centered = E_k - E_bar                              # (N, V)
    if topk_mask is not None:
        E_centered = E_centered.masked_fill(~topk_mask, 0.0)

    # Σ_k (E_k - Ē) F_k → (N, D)
    centered_force = (E_centered.unsqueeze(-1) * F_k).sum(dim=1)  # (N, D)
    dtau_dz = centered_force / (V_eff.unsqueeze(-1) * tau_col)    # (N, D)

    term2 = dL_dtau.unsqueeze(-1) * dtau_dz              # (N, D)

    # --- 合并 ---
    F_BE = (term1 + term2)                                # (N, D)

    # ================================================================
    #  Step 7: 放回 (B, S, D) 并归一化
    # ================================================================

    num_valid = float(max(N, 1))

    force_a = torch.zeros_like(z_pred)
    force_a[valid_mask] = F_BE / num_valid

    force_b = torch.zeros_like(z_pred)
    force_b[valid_mask] = F_target / num_valid

    # ================================================================
    #  Step 8: 诊断
    # ================================================================

    p_target_vals = p_k[idx_n, true_ids]
    loss_elegant = (E_target / D).mean().item()

    E_k_for_min = E_k.clone()
    E_k_for_min[idx_n, true_ids] = float('inf')
    E_nearest_wrong = E_k_for_min.min(dim=-1).values
    signed_gap = E_nearest_wrong - E_target
    negative_ratio = (signed_gap < 0).float().mean().item()

    info = {
        "tau": tau_per.mean().item(),
        "tau_std": tau_per.std().item(),
        "p_target_mean": p_target_vals.mean().item(),
        "p_target_min": p_target_vals.min().item(),
        "energy_gap_mean": signed_gap.abs().mean().item(),
        "negative_margin_ratio": negative_ratio,
        "loss_elegant": loss_elegant,
        "loss_be": loss_val,
        "effective_k": effective_k,
    }

    return loss_val, force_a, force_b, info
