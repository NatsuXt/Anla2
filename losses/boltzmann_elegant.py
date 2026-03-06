"""
保存位置: Anla/losses/boltzmann_elegant.py

Boltzmann-Elegant 损失函数 (v4.4)
=========================================================================

v4.3 → v4.4 变更:

    [1] τ 排除 target — 消除自反馈正循环
        旧: τ_n = std_k(E_k^{(n)})            — 包含 E_target
        新: τ_n = std({E_k^{(n)} : k ≠ tgt})  — 仅竞争者能量

        根因:
            训练推进中 E_target → 0, 而 E_wrong 保持高位, 导致 std(E)
            被 target 与 competitors 之间的"能量鸿沟"主导,
            而非竞争者之间的区分难度。τ 因此失控上升, softmax 变平,
            学习信号消失 — 模型因"做对了"而被惩罚。

            排除 target 后, τ 只度量竞争者之间的分散度:
              · 竞争者能量相近 (邻居混淆) → τ 小 → softmax 锐利 → 强梯度
              · 竞争者已被推开 → τ 自然反映有效区分力
            等价于: 温度描述热浴 (竞争者) 而非目标粒子

    [2] force_b 升级为 Term 1 Boltzmann 力 (含排斥)
        旧: force_b = F_target                 — 纯吸引力
        新: force_b = (1/τ)(F_target - Σ p_k F_k)  — 吸引 + 排斥

        排斥力使 embedding 可以直接推开竞争者, 而非仅靠模型权重间接实现。
        不含 τ-throughput 修正 (Term 2), 因为 embedding 更新不需要感知
        τ 对自身的依赖。

    [3] τ-throughput 求和排除 target
        dτ/dz* 的求和只经过竞争者, 切断 "z → E_target → τ" 自反馈路径。

核心公式:

    E_k(z) = Σ_d [ ln²(r_d / r̂_{k,d}) + |u_d - û_{k,d}|² ]

    τ_n = std({E_k^{(n)} : k ≠ target_n})   — per-sample 温度, 零超参

    p_k = softmax(-E_k / τ)                  — logits 仍包含所有 V 个 token

    L = -log p_target

完整 Wirtinger 梯度 (含 τ-throughput):

    dL/dz* = (1/τ)(F_target - Σ p_k F_k)
           + [(Ē_p - E_target) / ((V-1) · τ³)]
             · Σ_{k≠tgt} (E_k - Ē_c) F_k

    其中:
        F_k = ∂E_k/∂z*                — L_Elegant 流形力
        Ē_p = Σ p_k E_k               — Boltzmann 加权平均能量
        Ē_c = mean({E_k : k ≠ tgt})   — 竞争者算术平均能量
        V-1  = 竞争者数量

    第一项: 标准 Boltzmann-Elegant 力 (吸引 - 排斥)
    第二项: τ 对 z 的依赖产生的修正力 (仅通过竞争者传导)
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
    #  Step 4: τ_n = std({E_k : k ≠ target}) — per-sample 温度
    # ================================================================
    #
    # [v4.4] τ 排除 target:
    #   温度度量的是竞争者 (热浴) 的能量分散度, 不包含目标粒子。
    #   这消除了 "E_target→0 → std↑ → τ↑ → softmax变平" 的正反馈循环。
    #
    #   构造 competitor_mask: 在 topk_mask (如有) 基础上排除 target 位置。
    #   τ = std(E_competitors), Ē_c = mean(E_competitors)。

    # 构造竞争者掩码: 排除 target 位置
    competitor_mask = torch.ones(N, V, dtype=torch.bool, device=E_k.device)
    competitor_mask[idx_n, true_ids] = False                       # 排除 target

    if use_topk:
        # topk 模式下, 竞争者 = topk_set ∩ {k ≠ target}
        competitor_mask = competitor_mask & topk_mask

    # 用 nan-masking 安全计算 mean 和 std
    E_comp = E_k.masked_fill(~competitor_mask, float('nan'))       # (N, V)
    E_bar_c = torch.nanmean(E_comp, dim=-1, keepdim=True)         # (N, 1)
    tau_per = torch.nanmean(
        (E_comp - E_bar_c).pow(2), dim=-1
    ).sqrt() + eps                                                  # (N,)
    V_eff = competitor_mask.sum(dim=-1).float()                    # (N,) = V-1 或 topk-1

    tau_col = tau_per.unsqueeze(-1)                 # (N, 1) for broadcasting

    # ================================================================
    #  Step 5: Boltzmann 概率与损失
    # ================================================================
    #
    # 注: logits 仍包含所有 V 个 token (包括 target)。
    #     只有 τ 的计算排除了 target, loss/概率的定义不变。

    logits = -E_k / tau_col                         # (N, V)

    if use_topk:
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
    # Term 2: τ-throughput 修正 (仅通过竞争者传导)
    #   [v4.4] 求和排除 target, 切断 z → E_target → τ 自反馈
    #   = [(Ē_p - E_target) / ((V-1) · τ³)] · Σ_{k≠tgt} (E_k - Ē_c) F_k

    # --- Term 1 ---
    weighted_F = (p_k.unsqueeze(-1) * F_k).sum(dim=1)   # (N, D)
    term1 = (1.0 / tau_col) * (F_target - weighted_F)    # (N, D)

    # --- Term 2: τ-throughput (竞争者 only) ---
    # ∂L/∂τ = (Ē_p - E_target) / τ²
    E_p_mean = (p_k * E_k).sum(dim=-1)                   # (N,)
    dL_dtau = (E_p_mean - E_target) / (tau_per ** 2)      # (N,)

    # [v4.4] dτ/dz* = [1/((V-1) · τ)] Σ_{k≠tgt} (E_k - Ē_c) F_k
    #   E_bar_c 已在 Step 4 中基于竞争者计算
    E_centered = E_k - E_bar_c                             # (N, V)
    # 排除 target 和 topk 之外的 token
    E_centered = E_centered.masked_fill(~competitor_mask, 0.0)

    # Σ_{k≠tgt} (E_k - Ē_c) F_k → (N, D)
    centered_force = (E_centered.unsqueeze(-1) * F_k).sum(dim=1)  # (N, D)
    dtau_dz = centered_force / (V_eff.unsqueeze(-1) * tau_col)    # (N, D)

    term2 = dL_dtau.unsqueeze(-1) * dtau_dz              # (N, D)

    # --- 合并 ---
    F_BE = (term1 + term2)                                # (N, D)

    # ================================================================
    #  Step 7: 放回 (B, S, D) 并归一化
    # ================================================================

    num_valid = float(max(N, 1))

    # Path A: 完整 Boltzmann 力 (Term 1 + Term 2) → 模型权重
    force_a = torch.zeros_like(z_pred)
    force_a[valid_mask] = F_BE / num_valid

    # [v4.4] Path B: Term 1 Boltzmann 力 (吸引 + 排斥, 无 τ-throughput) → embedding
    #   旧: force_b = F_target (纯吸引)
    #   新: force_b = (1/τ)(F_target - Σ p_k F_k) (含排斥, 推开竞争者)
    force_b = torch.zeros_like(z_pred)
    force_b[valid_mask] = term1 / num_valid

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
