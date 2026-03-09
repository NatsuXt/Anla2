"""
保存位置: Anla/losses/boltzmann_cosine.py

Boltzmann-Cosine 损失函数
=========================================================================

将 Boltzmann 框架的能量函数从 L_Elegant 流形距离替换为复数余弦相似度。
保留 N 体力场的全部结构, 适用于多模态输出分布 (如 NLP)。

核心变更:
    旧 (L_Elegant): E_k = Σ_d [ln²(r/r̂) + |u−û|²]    — 位置匹配, O(D)
    新 (Cosine):    E_k = −cos_ℂ(z, w_k)               — 方向匹配, [-1,1]

    位置匹配要求 z 精确命中 w_target 的位置,
    方向匹配只要求 z 指向 w_target 的方向。
    NLP 中同一 token 在不同上下文中产生不同的 z,
    方向匹配天然支持这种多模态输出。

τ 稳定性:
    E_k ∈ [−1, 1] 恒定有界 → τ = O(1), 不可能发散。
    对比 L_Elegant: E_k = O(D), τ 在 NLP 上从 11 失控增长到 144。

Wirtinger 梯度 (Path A, 对 z*):
    F_k = ∂E_k/∂z* = (1/(2‖z‖)) · (s_k · ẑ − ŵ_k)
    dL/dz* = (1/τ) · (F_tgt − Σ p_k F_k)

Wirtinger 梯度 (Path B, 对 w_k*, 全 V 个 embedding):
    G_k = ∂E_k/∂w_k* = (1/(2‖w_k‖)) · (s_k · ŵ_k − ẑ)
    dL/dw_k* = [(δ_{k,tgt} − p_k) / τ] · G_k

    每个 k ∈ {0,...,V−1} 都有非零梯度:
    - k = target: 吸引力 (将 w_tgt 旋转向 z)
    - k ≠ target: 排斥力 (将 w_k 旋转离 z, 权重 ∝ p_k)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


def compute_boltzmann_cosine_loss_and_force(
    z_pred: torch.Tensor,
    all_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: Optional[int] = None,
) -> Tuple[float, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Boltzmann-Cosine 损失。方向匹配能量 + 自洽温度 + N 体力场。

    Parameters
    ----------
    z_pred : (B, S, D), complex
        模型预测的流形坐标
    all_embeds : (V, D), complex
        detached embedding 快照
    target_ids : (B, S), long
        目标 token ID, 非 mask 位置为 -100
    valid_mask : (B, S), bool
        有效位置掩码
    topk : int, optional
        Top-K 聚焦。None = 全 V。

    Returns
    -------
    loss_val : float
        标量损失值
    force_a : (B, S, D), complex
        Path A 梯度 (dL/dz*, 用于 Transformer 权重更新)
    emb_grad : (V, D), complex
        Path B 梯度 (dL/dw_k*, 全 V 个 embedding 的梯度)
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
        zero_emb = torch.zeros_like(all_embeds)
        return 0.0, zero_force, zero_emb, {
            "tau": 0, "tau_0": 0, "p_target_mean": 0, "effective_k": 0,
        }

    # ================================================================
    #  Step 1: 复数余弦相似度 s_k 与能量 E_k
    # ================================================================
    #
    # s_k = Re(z^H w_k) / (‖z‖ · ‖w_k‖)
    # E_k = −s_k ∈ [−1, 1]
    #
    # 展开: Re(z^H w) = Σ_d |z_d|·|w_d|·cos(φ_d − θ_d)
    # 编码了相位对齐 (cos) 和模长一致性 (|z||w| 加权)

    # 范数
    z_norm = z_valid.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=eps)  # (N, 1)
    w_norm = all_embeds.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=eps)  # (V, 1)

    # 单位向量
    z_hat = z_valid / z_norm                  # (N, D)
    w_hat = all_embeds / w_norm               # (V, D)

    # Hermitian 内积: z^H w = conj(z) · w
    # 批量: (N, D) @ (D, V) = (N, V) — 但需要 conj
    # Re(z^H W^T) = Re(conj(z) @ W^T) 其中 W 是 (V, D)
    # 对单位向量: Re(ẑ^H ŵ_k) = cos_ℂ(z, w_k)
    cos_sim = torch.real(z_hat @ w_hat.conj().T)     # (N, V)

    # 能量
    E_k = -cos_sim                                     # (N, V), ∈ [-1, 1]

    # ================================================================
    #  Step 2: F_k — Path A 力 (N, V, D)
    # ================================================================
    #
    # F_k = ∂E_k/∂z* = −∂s_k/∂z*
    #     = (1/(2‖z‖)) · (s_k · ẑ − ŵ_k)
    #
    # 推导见设计文档 boltzmann_cosine_design.md §3.1

    inv_2_z_norm = 0.5 / z_norm               # (N, 1)

    # s_k · ẑ: (N, V) × (N, D) → (N, V, D)
    # ŵ_k: (V, D) → broadcast (1, V, D)
    F_k = inv_2_z_norm.unsqueeze(1) * (
        cos_sim.unsqueeze(-1) * z_hat.unsqueeze(1)   # (N, V, D)
        - w_hat.unsqueeze(0)                           # (1, V, D)
    )                                                  # (N, V, D)

    # ================================================================
    #  Step 3: Target 量
    # ================================================================

    idx_n = torch.arange(N, device=z_pred.device)
    E_target = E_k[idx_n, true_ids]                    # (N,)
    F_target = F_k[idx_n, true_ids]                    # (N, D)

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
    #  Step 4: 自洽 τ (单步, 算法不变)
    # ================================================================
    #
    # E_k ∈ [−1, 1] → std(E_k) ≤ 1 → τ = O(1), 不可能发散

    competitor_mask = torch.ones(N, V, dtype=torch.bool, device=E_k.device)
    competitor_mask[idx_n, true_ids] = False

    if use_topk:
        competitor_mask = competitor_mask & topk_mask

    # Step A: τ₀ = std(E_competitors)
    E_comp = E_k.masked_fill(~competitor_mask, float('nan'))
    E_bar_c = torch.nanmean(E_comp, dim=-1, keepdim=True)
    tau_0_per = torch.nanmean(
        (E_comp - E_bar_c).pow(2), dim=-1
    ).sqrt() + eps                                              # (N,)

    tau_0_col = tau_0_per.unsqueeze(-1)                         # (N, 1)

    # Step B: p_k^(0) = softmax(-E_k / τ₀) 仅竞争者
    logits_for_tau = (-E_k / tau_0_col).masked_fill(
        ~competitor_mask, float('-inf'))
    p_tau = torch.softmax(logits_for_tau, dim=-1)

    # Step C: τ* = Boltzmann-weighted std
    E_bar_p = (p_tau * E_k).sum(dim=-1, keepdim=True)
    E_dev = (E_k - E_bar_p).masked_fill(~competitor_mask, 0.0)
    var_p = (p_tau * E_dev.pow(2)).sum(dim=-1)
    tau_star_per = var_p.sqrt() + eps                            # (N,)

    tau_star_col = tau_star_per.unsqueeze(-1)                    # (N, 1)

    # ================================================================
    #  Step 5: 最终 Boltzmann 概率与损失
    # ================================================================

    logits = -E_k / tau_star_col                                # (N, V)

    if use_topk:
        logits = logits.masked_fill(~topk_mask, float('-inf'))

    loss_per_sample = F.cross_entropy(logits, true_ids, reduction='none')
    loss_val = loss_per_sample.mean().item()

    p_k_final = torch.softmax(logits, dim=-1)                   # (N, V)

    # ================================================================
    #  Step 6: Path A — Wirtinger 梯度 dL/dz*
    # ================================================================
    #
    # dL/dz* = (1/τ*) · (F_tgt − Σ p_k F_k)

    weighted_F = (p_k_final.unsqueeze(-1) * F_k).sum(dim=1)    # (N, D)

    gradient_a = (F_target - weighted_F) / tau_star_col         # (N, D)

    num_valid = float(max(N, 1))

    force_a = torch.zeros_like(z_pred)
    force_a[valid_mask] = gradient_a / num_valid

    # ================================================================
    #  Step 7: Path B — 全词表 embedding 梯度 dL/dw_k*
    # ================================================================
    #
    # 对每个 embedding k, 聚合所有 N 个 masked 位置的梯度:
    #
    #   dL/dw_k* = (1/N) · Σ_n [(δ_{k,tgt_n} − p_k^(n)) / τ_n*]
    #              · (1/(2‖w_k‖)) · (s_k^(n) · ŵ_k − ẑ_n)
    #
    # 分解为两项:
    #   coeff_{n,k} = (δ_{k,tgt_n} − p_k^(n)) / τ_n*
    #   term1_k = ŵ_k · Σ_n coeff_{n,k} · s_k^(n)     — 径向 (沿 ŵ_k)
    #   term2_k = Σ_n coeff_{n,k} · ẑ_n                — 角度 (沿 ẑ_n)

    with torch.no_grad():
        # coeff: (N, V) — Boltzmann 梯度系数
        # δ_{k,tgt_n}: one-hot, 在 (n, tgt_n) 处为 1
        coeff = -p_k_final / tau_star_col                       # (N, V)  负的排斥部分
        coeff[idx_n, true_ids] += 1.0 / tau_star_per            # target 位置加吸引力

        # term1 系数: Σ_n coeff_{n,k} · s_k^(n) → (V,) 实数
        term1_scalar = (coeff * cos_sim).sum(dim=0)             # (V,)

        # term2: Σ_n coeff_{n,k} · ẑ_n → (V, D) 复数
        # coeff: (N, V) real, z_hat: (N, D) complex
        # (N, V)^T @ (N, D) = (V, N) @ (N, D) = (V, D)
        term2 = coeff.to(z_hat.dtype).T @ z_hat                # (V, D)

        # inv_2_w_norm: (V, 1)
        inv_2_w_norm = 0.5 / w_norm                            # (V, 1)

        # emb_grad = (1/(2‖w_k‖·N)) · (ŵ_k · term1_scalar − term2)
        emb_grad = inv_2_w_norm * (
            w_hat * term1_scalar.unsqueeze(-1).to(w_hat.dtype)  # (V, D)
            - term2                                              # (V, D)
        ) / num_valid                                           # (V, D)

    # ================================================================
    #  Step 8: 诊断
    # ================================================================

    p_target_vals = p_k_final[idx_n, true_ids]

    # target 余弦相似度 (越高越好)
    cos_target = cos_sim[idx_n, true_ids]                       # (N,)

    # Energy gap
    E_k_for_min = E_k.clone()
    E_k_for_min[idx_n, true_ids] = float('inf')
    E_nearest_wrong = E_k_for_min.min(dim=-1).values
    signed_gap = E_nearest_wrong - E_target
    negative_ratio = (signed_gap < 0).float().mean().item()

    info = {
        # τ 相关 (O(1) 尺度, 有界)
        "tau": tau_star_per.mean().item(),
        "tau_std": tau_star_per.std().item(),
        "tau_0": tau_0_per.mean().item(),

        # Boltzmann 概率
        "p_target_mean": p_target_vals.mean().item(),
        "p_target_min": p_target_vals.min().item(),

        # 余弦相似度
        "cos_target_mean": cos_target.mean().item(),

        # 能量间距
        "energy_gap_mean": signed_gap.abs().mean().item(),
        "negative_margin_ratio": negative_ratio,

        # 损失
        "loss_be": loss_val,

        # 配置
        "effective_k": effective_k,
    }

    return loss_val, force_a, emb_grad, info
