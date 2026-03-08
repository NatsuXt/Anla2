"""
保存位置: Anla/losses/boltzmann_elegant.py

Boltzmann-Elegant 损失函数 (v5.2)
=========================================================================

v4.4 → v5.2 变更:

    [1] 能量归一化 — 消除维度依赖
        旧: E_k = Σ_d [ln²(r_d/r̂_{k,d}) + |u_d - û_{k,d}|²]       — O(D)
        新: Ẽ_k = (1/D) Σ_d [ln²(r_d/r̂_{k,d}) + |u_d - û_{k,d}|²] — O(1)

        根因:
            E_k 是 D 个维度的求和, 因此 E_k = O(D)。
            τ = std(E_k) = O(√D), 导致:
              · τ 的绝对值随 d_model 变化, 不同配置之间不可比
              · 高维空间中 τ 偏大, 引入虚假的维度诅咒
            归一化后 Ẽ_k = O(1), τ̃ = O(1/√D), 消除维度依赖。

    [2] 自洽 τ (单步) — 用 Boltzmann 概率自加权
        旧: τ = std_uniform({E_k : k ≠ target})
            所有竞争者等权重, 远处 token 主导 τ

        新: 两步流程 (不做迭代, 训练循环本身就是外层迭代):
            Step A: τ₀ = std_uniform({Ẽ_k : k ∈ C})           — 初始估计
            Step B: p_k^(0) = softmax(-Ẽ_k / τ₀)               — 初始概率
            Step C: Ē_p = Σ p_k^(0) Ẽ_k                       — Boltzmann 加权均值
            Step D: τ* = √[Σ p_k^(0) (Ẽ_k - Ē_p)²] + ε       — 自洽温度

        物理意义:
            τ* 度量的是 "有效竞争者群体" 内部的能量分散度,
            其中 "有效竞争者" 由 Boltzmann 分布自身定义。
            p_k 极小的远处 token 对 τ* 几乎无贡献 (被自动压制),
            只有 p_k 显著的真正竞争者参与温度计算。

            不需要在单步内求解自洽方程 τ = f(τ) 的不动点 ——
            训练循环本身就是不动点迭代的外层循环:
            每步重新计算 E_k → 重新计算 p_k → 重新计算 τ*,
            只要方向对了就够了。

            类比:
              · 平均场论: 用上一步的平均场计算当前步
              · EM 算法:  E-step 用旧参数, M-step 更新
              · TD Learning: 单步 Bellman 更新

    [3] 消除 Term 2 (τ-throughput) — 梯度简化
        旧: dL/dz* = Term1 + Term2
            Term2 = [(Ē_p - E_target) / ((V-1)·τ³)] · Σ_{k≠tgt}(E_k - Ē_c)F_k

        新: dL/dz* = Term1 only
            τ* 视为常数 (stop-gradient), 不参与梯度流

        理由:
            a) τ* 已通过 p_k 自适应 — 不需要梯度来教它变成什么值
            b) τ 稳定时 ∂τ/∂z 本身很小, Term 2 贡献微弱
            c) 去掉 Term 2 消除了远处 token 通过 τ-throughput 注入的梯度噪声
            d) 大幅简化代码和计算

        物理类比:
            热浴温度是环境条件, 粒子在给定温度下运动,
            不需要通过改变粒子位置来改变热浴温度。

    [4] force_a 与 force_b 统一为 Term 1
        旧: force_a = Term1 + Term2, force_b = Term1
        新: force_a = force_b = Term1 (因为 Term 2 已消除)

        但为了保持接口兼容和未来扩展空间, 仍然返回两个独立的 force。

核心公式 (v5.2):

    Ẽ_k(z) = (1/D) Σ_d [ ln²(r_d / r̂_{k,d}) + |u_d - û_{k,d}|² ]

    τ₀ = std({Ẽ_k : k ≠ target})                  — uniform-weighted
    p_k^(0) = softmax(-Ẽ_k / τ₀)                    — 初始 Boltzmann 分布
    τ* = √[Σ_{k≠tgt} p_k^(0) (Ẽ_k - Ē_p)²] + ε   — self-consistent 温度

    logits = -Ẽ_k / τ*                              — 最终 logits (含 target)
    p_k = softmax(logits)                            — 最终 Boltzmann 分布
    L = -log p_target

    dL/dz* = (1/(τ* · D)) (F_target - Σ p_k F_k)   — 唯一梯度项

    其中:
        F_k = ∂E_k/∂z*     — L_Elegant 流形力 (未归一化, 仍基于 E_k 而非 Ẽ_k)
        1/(τ*·D) 中的 D    — 来自 Ẽ = E/D 的链式法则: ∂L/∂z* = ∂L/∂Ẽ · ∂Ẽ/∂z*
                              = (1/τ*)(F_target - ΣpF) · (1/D)
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
    Boltzmann-Elegant 损失 (v5.2)。零超参, 能量归一化, 自洽温度。

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
    # 原始能量 E_k_raw = Σ_d [ln²(r_d/r̂_{k,d}) + |u_d - û_{k,d}|²]
    # 归一化能量 Ẽ_k = E_k_raw / D
    #
    # [v5.2] 归一化使得 Ẽ_k = O(1), 消除对 d_model 的依赖。

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

    # [v5.2] 归一化能量 (用于 τ、logits、loss)
    E_k = E_k_raw / D                                  # (N, V)

    # ================================================================
    #  Step 2: F_k — 流形力 (N, V, D)
    # ================================================================
    #
    # F_k = ∂E_k_raw/∂z* (基于未归一化的 E_k_raw)
    #
    # Wirtinger 导数:
    #   径向力: (ln(r/r̂) / r) · u
    #   切向力: (u·u·û* - û) / (2r)
    #
    # 注意: F_k 不除以 D, 因为链式法则中 1/D 因子
    #       在最终梯度组装时统一乘入。

    F_radial = (log_ratio / r) * u                     # (N, V, D)
    F_tangential = (u * u * u_hat.conj() - u_hat) / (2.0 * r)  # (N, V, D)
    F_k = F_radial + F_tangential                      # (N, V, D)

    # ================================================================
    #  Step 3: Target 量
    # ================================================================

    idx_n = torch.arange(N, device=z_pred.device)
    E_target = E_k[idx_n, true_ids]                    # (N,) 归一化能量
    F_target = F_k[idx_n, true_ids]                    # (N, D) 未归一化力

    # ================================================================
    #  Step 3.5: Top-K 聚焦排斥
    # ================================================================

    use_topk = (topk is not None) and (topk < V)
    effective_k = topk if use_topk else V

    if use_topk:
        # 基于归一化能量选 Top-K (排序不受 1/D 影响)
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
    # [v5.2] 两步流程:
    #
    #   Step A: τ₀ = std_uniform({Ẽ_k : k ∈ C})
    #           初始温度估计, 来自竞争者的 uniform-weighted std。
    #           这是 v4.4 的定义 (归一化后), 作为迭代的起点。
    #
    #   Step B: p_k^(0) = softmax(-Ẽ_k / τ₀)  (仅竞争者)
    #           用 τ₀ 计算初始 Boltzmann 分布。
    #
    #   Step C: τ* = √[Σ_{k∈C} p_k^(0) (Ẽ_k - Ē_p)²] + ε
    #           用 Boltzmann 分布自加权计算最终温度。
    #           远处 token 的 p_k 极小, 对 τ* 几乎无贡献。
    #
    #   不在单步内迭代到不动点 ——
    #   训练循环本身就是不动点迭代的外层循环。

    # ---- 构造竞争者掩码: 排除 target 位置 ----
    competitor_mask = torch.ones(N, V, dtype=torch.bool, device=E_k.device)
    competitor_mask[idx_n, true_ids] = False            # 排除 target

    if use_topk:
        # 竞争者 = topk_set ∩ {k ≠ target}
        competitor_mask = competitor_mask & topk_mask

    # ---- Step A: τ₀ = std_uniform(Ẽ_competitors) ----
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
    # Ē_p = Σ_{k∈C} p_k^(0) · Ẽ_k — Boltzmann 加权均值
    E_bar_p = (p_tau * E_k).sum(dim=-1, keepdim=True)          # (N, 1)

    # Var_p = Σ_{k∈C} p_k^(0) · (Ẽ_k - Ē_p)² — Boltzmann 加权方差
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
    # [v5.2] 只有 Term 1:
    #
    #   dL/dz* = (1 / (τ* · D)) · (F_target - Σ_k p_k F_k)
    #
    # 其中:
    #   · 1/τ* 来自 softmax(-Ẽ/τ*) 对 Ẽ 的导数
    #   · 1/D  来自 Ẽ = E/D 的链式法则 (F_k = ∂E/∂z*, Ẽ 对 z* 的导数是 F_k/D)
    #   · F_target - Σ p_k F_k 是吸引力减去 Boltzmann 期望排斥力

    # Σ_k p_k F_k — Boltzmann 加权排斥力
    weighted_F = (p_k_final.unsqueeze(-1) * F_k).sum(dim=1)    # (N, D)

    # 完整梯度 (τ* 视为常数)
    #   scale = 1 / (τ* · D)
    #   gradient = scale * (F_target - weighted_F)
    scale = 1.0 / (tau_star_col * D)                            # (N, 1)
    gradient = scale * (F_target - weighted_F)                  # (N, D)

    # ================================================================
    #  Step 7: 放回 (B, S, D) 并归一化
    # ================================================================

    num_valid = float(max(N, 1))

    # Path A: Boltzmann-Elegant 梯度 → 模型权重更新
    # [v5.2] 只有 Term 1 (τ* 视为常数)
    force_a = torch.zeros_like(z_pred)
    force_a[valid_mask] = gradient / num_valid

    # Path B: Boltzmann-Elegant 梯度 → embedding 直接更新
    # [v5.2] 与 Path A 相同 (Term 2 已消除)
    # 保持独立返回以维持接口兼容和未来扩展空间
    force_b = torch.zeros_like(z_pred)
    force_b[valid_mask] = gradient / num_valid

    # ================================================================
    #  Step 8: 诊断
    # ================================================================

    p_target_vals = p_k_final[idx_n, true_ids]                  # (N,)

    # loss_elegant: 归一化的 target 能量 (Ẽ_target, 已经是 E/D)
    loss_elegant = E_target.mean().item()

    # Energy gap (基于归一化能量)
    E_k_for_min = E_k.clone()
    E_k_for_min[idx_n, true_ids] = float('inf')
    E_nearest_wrong = E_k_for_min.min(dim=-1).values           # (N,)
    signed_gap = E_nearest_wrong - E_target                    # (N,)
    negative_ratio = (signed_gap < 0).float().mean().item()

    info = {
        # τ 相关 (均为归一化后的值)
        "tau": tau_star_per.mean().item(),                      # 自洽温度 (均值)
        "tau_std": tau_star_per.std().item(),                   # 自洽温度 (std)
        "tau_0": tau_0_per.mean().item(),                       # 初始温度 (uniform std)

        # Boltzmann 概率
        "p_target_mean": p_target_vals.mean().item(),
        "p_target_min": p_target_vals.min().item(),

        # 能量间距 (归一化后)
        "energy_gap_mean": signed_gap.abs().mean().item(),
        "negative_margin_ratio": negative_ratio,

        # 损失
        "loss_elegant": loss_elegant,                           # 归一化 target 能量
        "loss_be": loss_val,                                    # Boltzmann-Elegant 损失

        # 配置
        "effective_k": effective_k,
    }

    return loss_val, force_a, force_b, info
