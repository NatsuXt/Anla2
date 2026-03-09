"""
保存位置: Anla/losses/boltzmann_elegant.py

Boltzmann-Elegant 损失函数 (v7.1)
=========================================================================

v5.2 → v7.1 变更:

    [1] 取消能量归一化
        旧 (v5.2): Ẽ_k = (1/D) Σ_d [...]   — O(1)
        新 (v7.1): E_k = Σ_d [...]           — O(D)

        动机:
            v5.2 的归一化将 τ 压缩到 O(1/√D) ≈ 0.19,
            softmax 极尖锐。在确定性任务 (Ring Inpainting) 中可行,
            但在随机任务 (NLP byte-level MLM) 中,
            输出变异性导致 τ 自适应正反馈失控 (τ↑ → 梯度弱 → τ↑)。

            取消归一化后 τ = O(√D) ≈ 8–12,
            与 v4 的实验值 (τ=10.79) 和理论预测 (T* ∝ √D) 一致。

        梯度影响:
            旧: dL/dz* = (1/(τ*·D)) · (F_target - Σ p_k F_k)
            新: dL/dz* = (1/τ*) · (F_target - Σ p_k F_k)
            去掉了 1/D 因子 (不再需要 Ẽ = E/D 的链式法则)。

    以下 v5.2 特性保持不变:
    [v5.2.2] 自洽 τ (单步 Boltzmann 自加权)
    [v5.2.3] 消除 Term 2 (τ 视为常数, stop-gradient)
    [v5.2.4] force_a = force_b = Term 1

核心公式 (v7.1):

    E_k(z) = Σ_d [ ln²(r_d / r̂_{k,d}) + |u_d - û_{k,d}|² ]

    τ₀ = std({E_k : k ≠ target})
    p_k^(0) = softmax(-E_k / τ₀)
    τ* = √[Σ_{k≠tgt} p_k^(0) (E_k - Ē_p)²] + ε

    logits = -E_k / τ*
    p_k = softmax(logits)
    L = -log p_target

    dL/dz* = (1/τ*) · (F_target - Σ p_k F_k)
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
    Boltzmann-Elegant 损失 (v7.1)。零超参, 未归一化能量, 自洽温度。

    Parameters
    ----------
    z_pred : (B, S, D), complex
        模型预测的流形坐标
    all_embeds : (V, D), complex
        detached embedding 快照 (所有 vocab token)
    target_ids : (B, S), long
        目标 token ID, 非 MASK 位置为 -100
    valid_mask : (B, S), bool
        有效位置掩码 (即被 mask 的位置)
    topk : int, optional
        Top-K 聚焦排斥。None = 全 V (默认)。

    Returns
    -------
    loss_val : float
        标量损失值
    force_a : (B, S, D), complex
        Path A 的 Wirtinger 梯度 (用于模型权重更新)
    force_b : (B, S, D), complex
        Path B 的 Wirtinger 梯度 (用于 embedding 直接更新)
    info : dict
        诊断信息
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
            "tau": 0, "tau_0": 0, "p_target_mean": 0, "effective_k": 0,
        }

    # ================================================================
    #  Step 1: E_k — 流形能量 (N, V)
    # ================================================================
    #
    # E_k = Σ_d [ln²(r_d/r̂_{k,d}) + |u_d - û_{k,d}|²]   — O(D)
    #
    # [v7.1] 直接使用未归一化能量, 不除以 D。

    z_exp = z_valid.unsqueeze(1)              # (N, 1, D)
    e_exp = all_embeds.unsqueeze(0)           # (1, V, D)

    r = z_exp.abs() + eps                     # (N, 1, D) 预测的模长
    r_hat = e_exp.abs() + eps                 # (1, V, D) embedding 的模长
    u = z_exp / r                             # (N, 1, D) 预测的单位相位
    u_hat = e_exp / r_hat                     # (1, V, D) embedding 的单位相位

    # 径向能量: ln²(r/r̂) — 对数模长比
    log_ratio = torch.log(r) - torch.log(r_hat)   # (N, V, D)
    energy_mag = log_ratio.pow(2)                  # (N, V, D)

    # 相位能量: |u - û|² — 单位相位差
    energy_phase = (u - u_hat).abs().pow(2)        # (N, V, D)

    # 原始总能量 (未归一化, 用于计算力 F_k)
    E_k_raw = (energy_mag + energy_phase).sum(dim=-1)  # (N, V)

    # [v7.1] 取消能量归一化 — 直接使用 E_k = E_k_raw
    #
    # v5.2 引入 Ẽ_k = E_k_raw / D 以消除维度依赖,
    # 但这将 τ 压缩到 O(1/√D) 的范围 (~0.19),
    # 在 NLP 等随机任务中 softmax 过于尖锐,
    # 导致 τ 自适应正反馈失控 (τ↑ → 梯度弱 → τ↑)。
    #
    # 取消归一化后 E_k = O(D), τ = O(√D),
    # 对 D=64 预期 τ ≈ 8–12, 与 v4 的 τ=10.79 一致,
    # 也与 Springer (2025) 的理论预测 T* ∝ √D 一致。
    E_k = E_k_raw                                          # (N, V)

    # ================================================================
    #  Step 2: F_k — 流形力 (N, V, D)
    # ================================================================
    #
    # F_k = ∂E_k/∂z*
    #
    # Wirtinger 导数:
    #   径向力: (ln(r/r̂) / r) · u
    #   切向力: (u·u·û* - û) / (2r)

    F_radial = (log_ratio / r) * u                     # (N, V, D)
    F_tangential = (u * u * u_hat.conj() - u_hat) / (2.0 * r)  # (N, V, D)
    F_k = F_radial + F_tangential                      # (N, V, D)

    # ================================================================
    #  Step 3: Target 量
    # ================================================================

    idx_n = torch.arange(N, device=z_pred.device)
    E_target = E_k[idx_n, true_ids]                    # (N,) 能量
    F_target = F_k[idx_n, true_ids]                    # (N, D) 力

    # ================================================================
    #  Step 3.5: Top-K 聚焦排斥
    # ================================================================

    use_topk = (topk is not None) and (topk < V)
    effective_k = topk if use_topk else V

    if use_topk:
        # 基于能量选 Top-K (能量最低的 K 个)
        _, topk_idx = E_k.topk(topk, dim=-1, largest=False)
        topk_mask = torch.zeros(N, V, dtype=torch.bool, device=E_k.device)
        topk_mask.scatter_(1, topk_idx, True)
        # 确保 target 始终在 Top-K 中
        topk_mask[idx_n, true_ids] = True
    else:
        topk_mask = None

    # ================================================================
    #  Step 4: 自洽 τ (单步) — per-sample 温度
    # ================================================================
    #
    # 两步流程:
    #   Step A: τ₀ = std_uniform({E_k : k ∈ C})
    #   Step B: p_k^(0) = softmax(-E_k / τ₀)  (仅竞争者)
    #   Step C: τ* = √[Σ_{k∈C} p_k^(0) (E_k - Ē_p)²] + ε

    # ---- 构造竞争者掩码: 排除 target 位置 ----
    competitor_mask = torch.ones(N, V, dtype=torch.bool, device=E_k.device)
    competitor_mask[idx_n, true_ids] = False            # 排除 target

    if use_topk:
        # 竞争者 = topk_set ∩ {k ≠ target}
        competitor_mask = competitor_mask & topk_mask

    # ---- Step A: τ₀ = std_uniform(E_competitors) ----
    # 用 nan-masking 安全计算 mean 和 std
    E_comp = E_k.masked_fill(~competitor_mask, float('nan'))   # (N, V)
    E_bar_c = torch.nanmean(E_comp, dim=-1, keepdim=True)     # (N, 1)
    tau_0_per = torch.nanmean(
        (E_comp - E_bar_c).pow(2), dim=-1
    ).sqrt() + eps                                              # (N,)

    tau_0_col = tau_0_per.unsqueeze(-1)                         # (N, 1)

    # ---- Step B: p_k^(0) — 初始 Boltzmann 分布 (仅竞争者) ----
    # 此处的 softmax 只用于计算 τ*, 不用于最终 loss。
    # 只在竞争者上做 softmax (target 不参与温度计算)。
    logits_for_tau = (-E_k / tau_0_col).masked_fill(
        ~competitor_mask, float('-inf')
    )                                                           # (N, V)
    p_tau = torch.softmax(logits_for_tau, dim=-1)               # (N, V)

    # ---- Step C: τ* = Boltzmann-weighted std ----
    # Ē_p = Σ_{k∈C} p_k^(0) · E_k — Boltzmann 加权均值
    E_bar_p = (p_tau * E_k).sum(dim=-1, keepdim=True)          # (N, 1)

    # Var_p = Σ_{k∈C} p_k^(0) · (E_k - Ē_p)² — Boltzmann 加权方差
    E_dev = (E_k - E_bar_p).masked_fill(~competitor_mask, 0.0) # (N, V)
    var_p = (p_tau * E_dev.pow(2)).sum(dim=-1)                 # (N,)
    tau_star_per = var_p.sqrt() + eps                            # (N,)

    tau_star_col = tau_star_per.unsqueeze(-1)                    # (N, 1)

    # ================================================================
    #  Step 5: 最终 Boltzmann 概率与损失 (使用 τ*)
    # ================================================================
    #
    # logits 包含所有 V 个 token (包括 target)。
    # 温度使用 τ* (自洽温度)。

    logits = -E_k / tau_star_col                                # (N, V)

    if use_topk:
        logits = logits.masked_fill(~topk_mask, float('-inf'))

    loss_per_sample = F.cross_entropy(logits, true_ids, reduction='none')
    loss_val = loss_per_sample.mean().item()

    p_k_final = torch.softmax(logits, dim=-1)                   # (N, V)

    # ================================================================
    #  Step 6: Wirtinger 梯度 (τ* 视为常数, 无 Term 2)
    # ================================================================
    #
    # [v7.1] 取消归一化后:
    #
    #   dL/dz* = (1/τ*) · (F_target - Σ_k p_k F_k)
    #
    # 其中:
    #   · 1/τ* 来自 softmax(-E/τ*) 对 E 的导数
    #   · F_target - Σ p_k F_k 是吸引力减去 Boltzmann 期望排斥力

    # Σ_k p_k F_k — Boltzmann 加权排斥力
    weighted_F = (p_k_final.unsqueeze(-1) * F_k).sum(dim=1)    # (N, D)

    # 完整梯度 (τ* 视为常数)
    #   [v7.1] 取消归一化后, 不再有 1/D 链式法则因子
    #   scale = 1 / τ*
    #   gradient = scale * (F_target - weighted_F)
    scale = 1.0 / tau_star_col                               # (N, 1)
    gradient = scale * (F_target - weighted_F)                # (N, D)

    # ================================================================
    #  Step 7: 放回 (B, S, D) 并归一化
    # ================================================================

    num_valid = float(max(N, 1))

    # Path A: Boltzmann-Elegant 梯度 → 模型权重更新
    force_a = torch.zeros_like(z_pred)
    force_a[valid_mask] = gradient / num_valid

    # Path B: Boltzmann-Elegant 梯度 → embedding 直接更新
    force_b = torch.zeros_like(z_pred)
    force_b[valid_mask] = gradient / num_valid

    # ================================================================
    #  Step 8: 诊断
    # ================================================================

    p_target_vals = p_k_final[idx_n, true_ids]                  # (N,)

    # loss_elegant: target 能量均值
    loss_elegant = E_target.mean().item()

    # Energy gap
    E_k_for_min = E_k.clone()
    E_k_for_min[idx_n, true_ids] = float('inf')
    E_nearest_wrong = E_k_for_min.min(dim=-1).values           # (N,)
    signed_gap = E_nearest_wrong - E_target                    # (N,)
    negative_ratio = (signed_gap < 0).float().mean().item()

    info = {
        # τ 相关 (未归一化尺度, O(√D))
        "tau": tau_star_per.mean().item(),
        "tau_std": tau_star_per.std().item(),
        "tau_0": tau_0_per.mean().item(),

        # Boltzmann 概率
        "p_target_mean": p_target_vals.mean().item(),
        "p_target_min": p_target_vals.min().item(),

        # 能量间距
        "energy_gap_mean": signed_gap.abs().mean().item(),
        "negative_margin_ratio": negative_ratio,

        # 损失
        "loss_elegant": loss_elegant,                           # target 能量均值
        "loss_be": loss_val,                                    # Boltzmann-Elegant 损失

        # 配置
        "effective_k": effective_k,
    }

    return loss_val, force_a, force_b, info
