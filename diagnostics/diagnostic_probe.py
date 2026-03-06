"""
保存位置: Anla/diagnostics/diagnostic_probe.py

Anla 训练瓶颈诊断探针
=========================================================================

在 capacity_pressure_test 的训练循环中插入, 或在训练后单独运行。
输出六组诊断指标, 每组对应一个瓶颈假设:

    H1. 分类余量 (Margin) — L_Elegant 是否缺乏对比信号?
    H2. Attention 路由精度 — 注意力是否学会精确选源?
    H3. Block 有效贡献 — 单层 Transformer 是否在工作?
    H4. Embedding 几何 — 嵌入空间的拓扑与分离度
    H5. Moving target — Path B 是否导致目标震荡?
    H6. 混淆结构 — 错误预测落在环上的什么位置?

用法 (三种方式):

    1. CLI — 从 checkpoint 加载模型运行诊断:
       python -m Anla.diagnostics.diagnostic_probe --config A
       python -m Anla.diagnostics.diagnostic_probe --config A --checkpoint path/to/best_checkpoint.pth

    2. 训练结束后内联调用 (推荐):
       from Anla.diagnostics.diagnostic_probe import run_diagnostic_with_model, print_diagnostic_report
       report = run_diagnostic_with_model(model, gen, cfg, device)
       print_diagnostic_report(report)

    3. 训练循环内逐步调用 (轻量):
       from Anla.diagnostics.diagnostic_probe import probe_embedding_stability
       snap_prev = model.embedding.weight.data.clone()
       # ... 训练一步 ...
       snap_curr = model.embedding.weight.data.clone()
       stability = probe_embedding_stability(snap_prev, snap_curr, vocab_size)

修正记录 (相对于初版):
    [Fix #1] checkpoint 路径与 capacity_pressure_test.py 的保存约定一致:
             训练保存到 config_{name}/best_checkpoint.pth,
             初版误用 {name}/best_checkpoint.pth, 导致找不到文件。
    [Fix #2] 增加多路径回退搜索, 兼容不同目录结构。
    [Fix #3] 诊断报告保存到与 checkpoint 同目录, 而非创建新目录。
    [Fix #4] 增加 run_diagnostic_with_model() 接口, 支持训练循环内
             直接传入模型对象, 无需 checkpoint 文件。
"""

import argparse
import math
import os
import sys
import json
import glob
import time
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

# [Path Fix] 文件位置: Anla/diagnostics/diagnostic_probe.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# =====================================================================
#  H1: 分类余量 (Margin Analysis)
# =====================================================================
#
#  核心问题: L_Elegant 只有吸引力 (pull z_pred → z_target),
#            没有排斥力 (push z_pred ← z_wrong)。
#            Cross-entropy 天然具有排斥力 (降低错误类的 logit),
#            但 L_Elegant 不做分类, 因此没有这个机制。
#
#  如果 margin 很小: 即使 z_pred 靠近 z_target,
#            最近邻分类仍可能选错 (尤其当多个 token embedding 挤在一起时)。
#
#  健康标准: mean_margin > 0.3, negative_ratio < 5%

@torch.no_grad()
def probe_classification_margin(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_size: int,
) -> Dict[str, float]:
    """
    计算分类余量: margin = dist(z_pred, nearest_wrong) - dist(z_pred, target)

    margin > 0 → 正确分类
    margin < 0 → 错误分类
    margin ≈ 0 → 脆弱, 稍有扰动就会选错

    Parameters
    ----------
    model : nn.Module
        AnlaManifoldInpainter 实例
    input_ids : torch.Tensor
        输入序列, shape (Batch, Seq), 包含 MASK token
    target_ids : torch.Tensor
        目标序列, shape (Batch, Seq), 非 MASK 位置为 -100
    vocab_size : int
        词表大小 (不含 MASK token)

    Returns
    -------
    Dict 包含:
        margin_mean, margin_median, margin_std, margin_min:
            余量统计
        negative_ratio:
            margin < 0 的比例 (= 分类错误率)
        zero_margin_ratio:
            margin < 0.01 的比例 (= 脆弱预测率)
        confusion_ring_dist_mean, confusion_ring_dist_mode:
            错误预测与正确答案的环距离 (越小说明混淆越集中在近邻)
    """
    model.eval()

    # 前向推理, 得到所有位置的预测
    z_pred = model.forward(input_ids)

    # 获取完整词表的 embedding (不含 MASK token)
    all_embeds = model.embedding.weight.data[:vocab_size]  # (V, D)

    # 筛选有效的 MASK 位置
    valid_mask = (target_ids != -100)
    if not valid_mask.any():
        return {"margin_mean": 0.0, "margin_std": 0.0, "negative_ratio": 1.0,
                "n_samples": 0}

    z_masked = z_pred[valid_mask]           # (N, D), N = 有效 MASK 位置数
    true_ids = target_ids[valid_mask]       # (N,)

    # 距离矩阵: 每个预测到每个词表元素的 L2 距离平方
    # (N, V) = |z_masked - all_embeds|²
    dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(dim=-1)

    # 到正确 target 的距离
    dist_to_target = dists[torch.arange(len(true_ids)), true_ids]  # (N,)

    # 到最近 wrong token 的距离
    # 先把正确位置的距离设为 inf, 这样 min 一定不选到它
    dists_wrong = dists.clone()
    dists_wrong[torch.arange(len(true_ids)), true_ids] = float('inf')
    dist_to_nearest_wrong = dists_wrong.min(dim=-1).values  # (N,)

    # 余量 = dist_wrong - dist_target
    # 正值 → 正确分类, 负值 → 错误分类
    margin = dist_to_nearest_wrong - dist_to_target

    # 分析混淆结构: 最近的错误 token 是谁?
    nearest_wrong_ids = dists_wrong.argmin(dim=-1)  # (N,)

    # 错误 token 与正确 token 在环上的距离
    ring_dist_to_wrong = torch.min(
        (nearest_wrong_ids - true_ids).abs() % vocab_size,
        (true_ids - nearest_wrong_ids).abs() % vocab_size,
    ).float()

    return {
        "margin_mean": margin.mean().item(),
        "margin_median": margin.median().item(),
        "margin_std": margin.std().item(),
        "margin_min": margin.min().item(),
        "negative_ratio": (margin < 0).float().mean().item(),
        "zero_margin_ratio": (margin.abs() < 0.01).float().mean().item(),
        "confusion_ring_dist_mean": ring_dist_to_wrong.mean().item(),
        "confusion_ring_dist_mode": ring_dist_to_wrong.mode().values.item(),
        "n_samples": len(margin),
    }


# =====================================================================
#  H2: Attention 路由精度
# =====================================================================
#
#  核心问题: 对于 MASK 位置, Attention 应该集中在少数有信息的
#            context 位置上 (环上的邻居)。如果 Attention 是弥散的,
#            MASK 位置只能看到所有 token 的模糊平均。
#
#  健康标准: entropy_normalized < 0.6, top1_prob > 0.2

@torch.no_grad()
def probe_attention_routing(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask_id: int,
) -> Dict[str, float]:
    """
    分析 Attention 的路由模式: 熵, 集中度, 是否看对了位置。

    不依赖 training 模式的缓存, 而是手动计算到 attention weights,
    确保 eval 模式下也能工作。

    Parameters
    ----------
    model : nn.Module
        AnlaManifoldInpainter 实例
    input_ids : torch.Tensor
        输入序列, shape (Batch, Seq)
    target_ids : torch.Tensor
        目标序列 (用于确认 MASK 位置)
    mask_id : int
        MASK token 的 ID (= vocab_size)

    Returns
    -------
    Dict 包含:
        attn_entropy_at_mask_mean:
            MASK 位置的 Attention 熵均值 (越小越集中)
        attn_entropy_normalized:
            归一化熵 (除以 log(seq_len)), 0=完全集中, 1=均匀
        attn_top1_prob_mean:
            Top-1 位置的概率均值 (越大越集中)
        attn_top1_is_context_ratio:
            Top-1 位置是 context (非 MASK) 的比例
    """
    model.eval()

    # 手动计算到 attention weights (不依赖 training 缓存)
    # 复现前向路径: Embedding → Rotary → Norm1 → Q/K 投影 → Score
    z = model.embedding.forward(input_ids)
    z = model.rotary.forward(z)

    attn_layer = model.block.attn
    norm_z = model.block.norm1.forward(z)

    # Q, K 投影 + 多头拆分
    q = attn_layer._split_heads(attn_layer.w_q(norm_z))   # (B, H, S, Hd)
    k = attn_layer._split_heads(attn_layer.w_k(norm_z))   # (B, H, S, Hd)

    # Interference: S = Q @ K^H
    k_H = k.transpose(-1, -2).conj()
    scores = torch.matmul(q, k_H)                          # (B, H, S, S)

    # MagPhaseSoftmax 的 softmax 部分 (只取模长)
    scale = 1.0 / math.sqrt(attn_layer.head_dim)
    mag = torch.abs(scores)
    mag_probs = torch.softmax(mag * scale, dim=-1)          # (B, H, S, S)

    # 找到 MASK 位置
    mask_positions = (input_ids == mask_id)                  # (B, S)

    if not mask_positions.any():
        return {"attn_entropy_at_mask_mean": -1.0,
                "note": "no mask positions found"}

    B, H, Sq, Sk = mag_probs.shape

    entropy_list = []
    top1_prob_list = []
    top1_is_context_list = []

    for b in range(B):
        mask_pos = mask_positions[b].nonzero(as_tuple=True)[0]
        for pos in mask_pos:
            for h in range(H):
                probs = mag_probs[b, h, pos]  # (Sk,)

                # 信息熵: H = -Σ p_j log(p_j)
                log_probs = torch.log(probs + 1e-10)
                ent = -(probs * log_probs).sum().item()
                entropy_list.append(ent)

                # Top-1 概率
                top1_prob = probs.max().item()
                top1_prob_list.append(top1_prob)

                # Top-1 位置是否是 context token (非 MASK)?
                top1_pos = probs.argmax().item()
                top1_is_context = (input_ids[b, top1_pos] != mask_id).item()
                top1_is_context_list.append(float(top1_is_context))

    max_entropy = math.log(Sk)  # 均匀分布的熵

    return {
        "attn_entropy_at_mask_mean": float(np.mean(entropy_list)),
        "attn_entropy_at_mask_std": float(np.std(entropy_list)),
        "attn_entropy_max_possible": max_entropy,
        "attn_entropy_normalized": float(np.mean(entropy_list)) / max_entropy,
        "attn_top1_prob_mean": float(np.mean(top1_prob_list)),
        "attn_top1_is_context_ratio": float(np.mean(top1_is_context_list)),
        "n_attn_samples": len(entropy_list),
    }


# =====================================================================
#  H3: Block 有效贡献
# =====================================================================
#
#  核心问题: 如果 Residual 远大于 block 输出, 说明 Transformer block
#            几乎没在做有效计算 — 输出约等于输入的 embedding+rotary。
#
#  健康标准: attn_ratio > 0.1, ffn_ratio > 0.1

@torch.no_grad()
def probe_block_contribution(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Dict[str, float]:
    """
    测量 Attention 和 FFN 子层的输出相对于 Residual 的大小。

    手动逐步执行前向, 分别测量每个子层的输出幅度。

    Parameters
    ----------
    model : nn.Module
        AnlaManifoldInpainter 实例
    input_ids : torch.Tensor
        输入序列

    Returns
    -------
    Dict 包含:
        attn_to_input_ratio:
            |attn_output| / |input|, 越大说明 Attention 贡献越大
        ffn_to_input_ratio:
            |ffn_output| / |input|, 越大说明 FFN 贡献越大
        attn_phase_shift_mean:
            Attention 输出与输入的平均相位差 (rad)
    """
    model.eval()

    block = model.block

    # Embedding → Rotary → 得到 block 的输入
    z = model.embedding.forward(input_ids)
    z = model.rotary.forward(z)

    # --- 手动执行 Block 内部 ---

    # Attention 子层
    norm_z = block.norm1.forward(z)
    attn_out = block.attn.forward(norm_z)
    z_after_attn = z + attn_out  # Residual

    # FFN 子层
    norm_z2 = block.norm2.forward(z_after_attn)
    ff_h = block.ff1.forward(norm_z2)
    ff_h = block.act.forward(ff_h)
    ff_out = block.ff2.forward(ff_h)
    z_final = z_after_attn + ff_out  # Residual

    # --- 计算比值 ---
    input_norm = z.abs().mean().item()
    attn_out_norm = attn_out.abs().mean().item()
    ffn_out_norm = ff_out.abs().mean().item()
    final_norm = z_final.abs().mean().item()

    # Attention 输出与输入的相位差
    # 如果 Attention 只是复制输入, 相位差 ≈ 0
    phase_shift_attn = torch.angle(attn_out) - torch.angle(z)
    # wrap 到 [-π, π]
    phase_shift_attn = (phase_shift_attn + math.pi) % (2 * math.pi) - math.pi

    return {
        "input_mean_mag": input_norm,
        "attn_out_mean_mag": attn_out_norm,
        "ffn_out_mean_mag": ffn_out_norm,
        "final_mean_mag": final_norm,
        "attn_to_input_ratio": attn_out_norm / (input_norm + 1e-10),
        "ffn_to_input_ratio": ffn_out_norm / (input_norm + 1e-10),
        "attn_phase_shift_mean": phase_shift_attn.abs().mean().item(),
        "attn_phase_shift_std": phase_shift_attn.std().item(),
    }


# =====================================================================
#  H4: Embedding 几何结构
# =====================================================================
#
#  核心问题: NN% 只看 k=1, 但模加法需要更广泛的结构。
#            还需要知道: embedding 是否坍缩? 是否分离度足够?
#
#  健康标准: ring_knn_1 > 0.7, min_inter_dist > 0.01

@torch.no_grad()
def probe_embedding_geometry(
    all_embeds: torch.Tensor,
    vocab_size: int,
) -> Dict[str, Any]:
    """
    分析 embedding 空间的几何结构。

    Parameters
    ----------
    all_embeds : torch.Tensor
        完整 embedding 矩阵, shape (V+1, D) (包含 MASK token)
    vocab_size : int
        实际词表大小 (不含 MASK)

    Returns
    -------
    Dict 包含:
        ring_knn_{k}: k=1,2,3,5 的环近邻一致率
        nn_dist_*:    最近邻距离统计
        mag_*:        模长分布统计
        phase_diff_consistency_*: 相位差有序性
        n_consistent_dims: 相位差 std < 0.5 的维度数 (= 有效编码环结构的维度)
    """
    w = all_embeds[:vocab_size]  # (V, D)
    V, D = w.shape

    # 1. 两两距离矩阵 (复数展开为实虚拼接, 使用 L2 距离)
    w_real = torch.cat([w.real, w.imag], dim=-1)  # (V, 2D)
    dist_mat = torch.cdist(w_real, w_real)         # (V, V)
    dist_mat.fill_diagonal_(float('inf'))          # 排除自身

    # 最近邻距离统计
    nn_dists = dist_mat.min(dim=-1).values  # (V,)

    # 2. K-近邻环一致率 (k=1,2,3,5)
    #    检查 embedding 空间中的 k 个最近邻是否是环上的 k-邻域
    knn_results = {}
    for k in [1, 2, 3, 5]:
        _, topk_idx = dist_mat.topk(k, dim=-1, largest=False)  # (V, k)
        correct_count = 0
        total = V * k
        for i in range(V):
            for j in range(k):
                neighbor = topk_idx[i, j].item()
                # 环上距离 = min(|i-j| mod V, |j-i| mod V)
                ring_dist = min(abs(neighbor - i) % V, abs(i - neighbor) % V)
                if ring_dist <= k:
                    correct_count += 1
        knn_results[f"ring_knn_{k}"] = correct_count / total

    # 3. 模长分布
    mags = w.abs()  # (V, D)
    mag_per_token = mags.mean(dim=-1)  # (V,) 每个 token 的平均模长

    # 4. 相位有序性: 按环顺序排列的相邻 token 相位差
    #    理想环: 每步转相同角度 → std ≈ 0
    phases = torch.angle(w)  # (V, D)

    # 相邻 token 的相位差 (V-1 个差值)
    phase_diffs = phases[1:] - phases[:-1]  # (V-1, D)
    # wrap 到 [-π, π]
    phase_diffs = (phase_diffs + math.pi) % (2 * math.pi) - math.pi

    # 环首尾连接的相位差
    wrap_diff = phases[0] - phases[-1]
    wrap_diff = (wrap_diff + math.pi) % (2 * math.pi) - math.pi
    phase_diffs_all = torch.cat([phase_diffs, wrap_diff.unsqueeze(0)], dim=0)  # (V, D)

    # 每个维度的相位差标准差
    # 越小 → 该维度的环编码越有序
    phase_diff_std_per_dim = phase_diffs_all.std(dim=0)  # (D,)

    return {
        **knn_results,
        "nn_dist_mean": nn_dists.mean().item(),
        "nn_dist_min": nn_dists.min().item(),
        "nn_dist_max": nn_dists.max().item(),
        "nn_dist_std": nn_dists.std().item(),
        "mag_mean": mag_per_token.mean().item(),
        "mag_std": mag_per_token.std().item(),
        "mag_min": mag_per_token.min().item(),
        "mag_max": mag_per_token.max().item(),
        "phase_diff_consistency_mean": phase_diff_std_per_dim.mean().item(),
        "phase_diff_consistency_best": phase_diff_std_per_dim.min().item(),
        "n_consistent_dims": int((phase_diff_std_per_dim < 0.5).sum().item()),
        "total_dims": D,
    }


# =====================================================================
#  H5: Moving Target 检测
# =====================================================================
#
#  核心问题: Path B 更新 target embedding, 使得下一步的 z_target 不同。
#            如果更新幅度大, 模型在追一个快速移动的目标。
#
#  需要在训练循环内连续两步测量, 此处提供辅助函数。
#  CLI 模式无法运行此探针 (需要两个时间步的快照)。

@torch.no_grad()
def probe_embedding_stability(
    emb_snap_prev: torch.Tensor,
    emb_snap_curr: torch.Tensor,
    vocab_size: int,
) -> Dict[str, float]:
    """
    比较两个时间步的 embedding 变化。

    用于训练循环内逐步调用:
        snap_prev = model.embedding.weight.data.clone()
        # ... 训练一步 ...
        snap_curr = model.embedding.weight.data.clone()
        stability = probe_embedding_stability(snap_prev, snap_curr, vocab_size)

    Parameters
    ----------
    emb_snap_prev : torch.Tensor
        前一步的 embedding, shape (V+1, D)
    emb_snap_curr : torch.Tensor
        当前步的 embedding, shape (V+1, D)
    vocab_size : int
        实际词表大小 (不含 MASK)

    Returns
    -------
    Dict 包含:
        emb_delta_mean: 绝对变化量均值
        emb_relative_delta_mean: 相对变化量均值
        emb_phase_delta_mean: 相位变化均值 (rad)
    """
    w_prev = emb_snap_prev[:vocab_size]
    w_curr = emb_snap_curr[:vocab_size]

    delta = (w_curr - w_prev).abs()
    delta_norm = delta.mean(dim=-1)  # (V,) 每个 token 的平均变化量

    base_norm = w_prev.abs().mean(dim=-1) + 1e-10
    relative_delta = delta_norm / base_norm

    # 相位变化
    phase_prev = torch.angle(w_prev)
    phase_curr = torch.angle(w_curr)
    phase_delta = (phase_curr - phase_prev + math.pi) % (2 * math.pi) - math.pi

    return {
        "emb_delta_mean": delta_norm.mean().item(),
        "emb_delta_max": delta_norm.max().item(),
        "emb_relative_delta_mean": relative_delta.mean().item(),
        "emb_phase_delta_mean": phase_delta.abs().mean().item(),
        "emb_phase_delta_max": phase_delta.abs().max().item(),
    }


# =====================================================================
#  H6: 混淆结构分析
# =====================================================================

@torch.no_grad()
def probe_confusion_structure(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_size: int,
) -> Dict[str, Any]:
    """
    分析错误预测的结构: 模型把哪些 token 搞混了?

    如果混淆集中在环邻居 (ring_dist=1,2): 拓扑正确但精度不够
    如果混淆随机分布: 模型没有学到有用的结构

    Parameters
    ----------
    model : nn.Module
        AnlaManifoldInpainter 实例
    input_ids, target_ids : torch.Tensor
        输入/目标序列
    vocab_size : int
        词表大小 (不含 MASK)

    Returns
    -------
    Dict 包含:
        accuracy: 最近邻分类准确率
        wrong_ring_dist_mean: 错误预测与正确答案的平均环距离
        ring_dist_{1,2,3_5,6_10,11_plus}: 各环距范围的错误占比
    """
    model.eval()
    z_pred = model.forward(input_ids)
    all_embeds = model.embedding.weight.data[:vocab_size]

    valid_mask = (target_ids != -100)
    if not valid_mask.any():
        return {"accuracy": 0.0, "n_wrong": 0, "n_total": 0}

    z_masked = z_pred[valid_mask]
    true_ids = target_ids[valid_mask]

    # 最近邻分类
    dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(dim=-1)
    pred_ids = dists.argmin(dim=-1)

    wrong_mask = (pred_ids != true_ids)

    if not wrong_mask.any():
        return {"accuracy": 1.0, "n_wrong": 0, "n_total": len(pred_ids)}

    wrong_pred = pred_ids[wrong_mask]
    wrong_true = true_ids[wrong_mask]

    # 环距离
    ring_dists = torch.min(
        (wrong_pred - wrong_true).abs() % vocab_size,
        (wrong_true - wrong_pred).abs() % vocab_size,
    )

    # 统计环距离分布
    dist_counter = Counter(ring_dists.cpu().numpy().tolist())
    total_wrong = len(ring_dists)

    # 分桶: 1, 2, 3-5, 6-10, 11+
    buckets = {
        "ring_dist_1": 0, "ring_dist_2": 0, "ring_dist_3_5": 0,
        "ring_dist_6_10": 0, "ring_dist_11_plus": 0,
    }
    for d, c in dist_counter.items():
        d = int(d)
        if d == 1:
            buckets["ring_dist_1"] += c
        elif d == 2:
            buckets["ring_dist_2"] += c
        elif d <= 5:
            buckets["ring_dist_3_5"] += c
        elif d <= 10:
            buckets["ring_dist_6_10"] += c
        else:
            buckets["ring_dist_11_plus"] += c

    # 归一化为比例
    for k in buckets:
        buckets[k] = buckets[k] / total_wrong

    return {
        "accuracy": 1.0 - wrong_mask.float().mean().item(),
        "n_wrong": total_wrong,
        "n_total": len(pred_ids),
        "wrong_ring_dist_mean": ring_dists.float().mean().item(),
        **buckets,
    }


# =====================================================================
#  综合诊断: 从模型对象运行
# =====================================================================

def run_diagnostic_with_model(
    model: nn.Module,
    data_gen,
    cfg: Dict[str, Any],
    device: torch.device,
    n_batches: int = 10,
) -> Dict[str, Any]:
    """
    [推荐接口] 直接传入已训练模型, 无需 checkpoint 文件。

    可在训练结束后直接调用:
        report = run_diagnostic_with_model(model, gen, cfg, device)
        print_diagnostic_report(report)

    Parameters
    ----------
    model : nn.Module
        已训练的 AnlaManifoldInpainter 实例
    data_gen : RingSpanDataGeneratorWithHoldout
        数据生成器 (同训练用的实例)
    cfg : Dict
        配置字典, 需包含:
            name, vocab_size, batch_size, max_span_length
    device : torch.device
        计算设备
    n_batches : int
        用于诊断的 batch 数量 (更多 → 更稳定但更慢)

    Returns
    -------
    Dict: 包含 H1~H6 各组诊断结果的嵌套字典
    """
    vocab_size = cfg["vocab_size"]
    batch_size = cfg["batch_size"]
    max_span = cfg["max_span_length"]
    mask_id = vocab_size  # MASK token ID = vocab_size

    model.eval()

    # 收集多 batch 的结果
    margin_results = []
    attn_results = []
    block_results = []
    confusion_results = []

    for _ in range(n_batches):
        input_ids, target_ids = data_gen.generate_train_batch(batch_size, max_span)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        margin_results.append(probe_classification_margin(
            model, input_ids, target_ids, vocab_size))
        attn_results.append(probe_attention_routing(
            model, input_ids, target_ids, mask_id))
        block_results.append(probe_block_contribution(model, input_ids))
        confusion_results.append(probe_confusion_structure(
            model, input_ids, target_ids, vocab_size))

    # 对多 batch 结果求平均
    def avg_dicts(dicts, keys=None):
        """对 dict 列表中的数值字段求均值"""
        if keys is None:
            keys = [k for k in dicts[0]
                    if isinstance(dicts[0][k], (int, float))]
        result = {}
        for k in keys:
            vals = [d[k] for d in dicts
                    if k in d and isinstance(d[k], (int, float))]
            if vals:
                result[k] = float(np.mean(vals))
        return result

    # Embedding 几何 (全局属性, 只需一次)
    all_embeds = model.embedding.weight.data
    geom = probe_embedding_geometry(all_embeds, vocab_size)

    report = {
        "config": cfg.get("name", "unknown"),
        "H1_margin": avg_dicts(margin_results),
        "H2_attention": avg_dicts(attn_results),
        "H3_block": avg_dicts(block_results),
        "H4_geometry": geom,
        "H5_stability": {"note": "需要训练循环内逐步调用 probe_embedding_stability"},
        "H6_confusion": avg_dicts(confusion_results),
    }

    return report


# =====================================================================
#  报告打印
# =====================================================================

def print_diagnostic_report(report: Dict[str, Any]):
    """格式化打印完整诊断报告"""

    print()
    print("=" * 72)
    print(f"  Anla 诊断报告 — Config: {report['config']}")
    print("=" * 72)

    # --- H1: 分类余量 ---
    m = report["H1_margin"]
    margin_ok = m.get("margin_mean", 0) > 0.3 and m.get("negative_ratio", 1) < 0.1
    verdict = "✅" if margin_ok else "❌"

    print(f"\n  {verdict} H1: 分类余量 (L_Elegant 是否缺乏对比信号?)")
    print(f"  {'—' * 60}")
    print(f"    margin_mean:       {m.get('margin_mean', 0):.4f}  (健康: > 0.3)")
    print(f"    margin_median:     {m.get('margin_median', 0):.4f}")
    print(f"    negative_ratio:    {m.get('negative_ratio', 0):.2%}  (健康: < 10%)")
    print(f"    zero_margin_ratio: {m.get('zero_margin_ratio', 0):.2%}  (margin < 0.01)")
    print(f"    confusion_ring_dist: {m.get('confusion_ring_dist_mean', 0):.1f}  "
          f"(众数: {m.get('confusion_ring_dist_mode', 0):.0f})")

    if m.get("confusion_ring_dist_mean", 999) < 3:
        print(f"    → 混淆集中在环邻居: 拓扑正确但精度不足")
    elif m.get("negative_ratio", 0) > 0.3:
        print(f"    → 大量分类错误: embedding 空间可能坍缩")

    # --- H2: Attention 路由 ---
    a = report["H2_attention"]
    entropy_norm = a.get("attn_entropy_normalized", 1)
    attn_ok = entropy_norm < 0.6 and a.get("attn_top1_prob_mean", 0) > 0.2
    verdict = "✅" if attn_ok else "❌"

    print(f"\n  {verdict} H2: Attention 路由精度")
    print(f"  {'—' * 60}")
    print(f"    entropy (normalized): {entropy_norm:.3f}  (健康: < 0.6, 均匀=1.0)")
    print(f"    top1_prob_mean:       {a.get('attn_top1_prob_mean', 0):.3f}  (健康: > 0.2)")
    print(f"    top1_is_context:      {a.get('attn_top1_is_context_ratio', 0):.2%}")

    if entropy_norm > 0.8:
        print(f"    → Attention 近乎均匀分布: 没有学到路由")
    elif entropy_norm > 0.6:
        print(f"    → Attention 略有集中但不够尖锐")

    # --- H3: Block 贡献 ---
    b = report["H3_block"]
    attn_ratio = b.get("attn_to_input_ratio", 0)
    ffn_ratio = b.get("ffn_to_input_ratio", 0)
    block_ok = attn_ratio > 0.05 and ffn_ratio > 0.05
    verdict = "✅" if block_ok else "❌"

    print(f"\n  {verdict} H3: Block 有效贡献 (单层 Transformer 在工作吗?)")
    print(f"  {'—' * 60}")
    print(f"    attn_out / input:   {attn_ratio:.4f}  (健康: > 0.05)")
    print(f"    ffn_out / input:    {ffn_ratio:.4f}  (健康: > 0.05)")
    print(f"    input_mag:          {b.get('input_mean_mag', 0):.4f}")
    print(f"    final_mag:          {b.get('final_mean_mag', 0):.4f}")
    print(f"    attn_phase_shift:   {b.get('attn_phase_shift_mean', 0):.4f} rad")

    if attn_ratio < 0.01:
        print(f"    → Attention 输出微弱: block 几乎不在工作")
    if ffn_ratio < 0.01:
        print(f"    → FFN 输出微弱: 非线性变换没有贡献")

    # --- H4: Embedding 几何 ---
    g = report["H4_geometry"]
    knn1 = g.get("ring_knn_1", 0)
    collapse = g.get("nn_dist_min", 0) < 0.01
    verdict = "✅" if knn1 > 0.7 and not collapse else "❌"

    print(f"\n  {verdict} H4: Embedding 几何结构")
    print(f"  {'—' * 60}")
    print(f"    ring_knn_1:         {knn1:.2%}  (健康: > 70%)")
    print(f"    ring_knn_2:         {g.get('ring_knn_2', 0):.2%}")
    print(f"    ring_knn_3:         {g.get('ring_knn_3', 0):.2%}")
    print(f"    ring_knn_5:         {g.get('ring_knn_5', 0):.2%}")
    print(f"    nn_dist_min:        {g.get('nn_dist_min', 0):.4f}  "
          f"(坍缩阈值: < 0.01)")
    print(f"    nn_dist_mean:       {g.get('nn_dist_mean', 0):.4f}")
    print(f"    mag range:          "
          f"[{g.get('mag_min', 0):.3f}, {g.get('mag_max', 0):.3f}]")
    print(f"    consistent_dims:    "
          f"{g.get('n_consistent_dims', 0)} / {g.get('total_dims', 0)}"
          f"  (相位差 std < 0.5 的维度)")

    if collapse:
        print(f"    → ⚠️  存在极近 embedding 对, 可能坍缩!")

    # --- H5: Moving Target ---
    h5 = report.get("H5_stability", {})
    if "note" in h5:
        print(f"\n  ℹ️  H5: Moving Target (需训练循环内调用)")
        print(f"  {'—' * 60}")
        print(f"    {h5['note']}")
    else:
        print(f"\n  📊 H5: Moving Target")
        print(f"  {'—' * 60}")
        for k, v in h5.items():
            print(f"    {k}: {v:.6f}")

    # --- H6: 混淆结构 ---
    c = report["H6_confusion"]
    print(f"\n  📊 H6: 混淆结构分析")
    print(f"  {'—' * 60}")
    print(f"    accuracy:           {c.get('accuracy', 0):.2%}")
    print(f"    wrong_ring_dist:    {c.get('wrong_ring_dist_mean', 0):.1f}")

    d1 = c.get("ring_dist_1", 0)
    d2 = c.get("ring_dist_2", 0)
    d35 = c.get("ring_dist_3_5", 0)
    d610 = c.get("ring_dist_6_10", 0)
    d11 = c.get("ring_dist_11_plus", 0)
    print(f"    错误分布:")
    print(f"      环距 1:   {d1:.1%}")
    print(f"      环距 2:   {d2:.1%}")
    print(f"      环距 3-5: {d35:.1%}")
    print(f"      环距 6-10:{d610:.1%}")
    print(f"      环距 11+: {d11:.1%}")

    if d1 + d2 > 0.6:
        print(f"    → 错误高度集中在近邻: 拓扑学习正确, 精度不够")
        print(f"      诊断: 缺乏对比信号 (H1) 或 Attention 精度不足 (H2)")
    elif d11 > 0.3:
        print(f"    → 大量远距离混淆: 拓扑结构未充分学习")

    # --- 总结 ---
    print(f"\n  {'=' * 60}")
    print(f"  综合判断:")

    issues = []
    if not margin_ok:
        issues.append("H1-余量不足")
    if not attn_ok:
        issues.append("H2-Attention弥散")
    if not block_ok:
        issues.append("H3-Block无贡献")
    if knn1 < 0.7:
        issues.append("H4-拓扑不完整")
    if collapse:
        issues.append("H4-embedding坍缩")

    if not issues:
        print(f"    ✅ 各项指标健康, 瓶颈可能在模型容量或训练轮数")
    else:
        print(f"    ❌ 主要问题: {', '.join(issues)}")
        if "H1-余量不足" in issues and "H2-Attention弥散" in issues:
            print(f"    → 建议: 考虑引入对比损失, 并增加注意力的尖锐度")
        elif "H1-余量不足" in issues:
            print(f"    → 建议: L_Elegant 缺乏排斥力, 考虑引入负样本或对比项")
        elif "H2-Attention弥散" in issues:
            print(f"    → 建议: Attention 未学到路由, 检查 Q/K 投影是否有效")
        elif "H3-Block无贡献" in issues:
            print(f"    → 建议: Transformer block 未工作, 可能需要更深层或更大 d_model")

    print()


# =====================================================================
#  Checkpoint 搜索逻辑
# =====================================================================

def find_checkpoint(
    output_dir: str,
    config_name: str,
    explicit_path: Optional[str] = None,
) -> Optional[str]:
    """
    [Fix #1, #2] 按优先级搜索 checkpoint 文件。

    搜索顺序:
        1. 用户显式指定的路径 (--checkpoint 参数)
        2. config_{name}/best_checkpoint.pth  (与 capacity_pressure_test.py 一致)
        3. {name}/best_checkpoint.pth          (旧版/手动保存的情况)
        4. config_{name}/*.pth                 (任意 .pth 文件)
        5. {name}/*.pth                        (任意 .pth 文件)

    Parameters
    ----------
    output_dir : str
        顶层输出目录 (如 Logs/capacity_pressure_test)
    config_name : str
        配置名 (如 A_v64_d64)
    explicit_path : str, optional
        用户通过 --checkpoint 显式指定的路径

    Returns
    -------
    str or None: 找到的 checkpoint 路径, 或 None
    """
    # 1. 显式路径
    if explicit_path is not None:
        if os.path.exists(explicit_path):
            return explicit_path
        else:
            print(f"  ⚠️  指定的 checkpoint 不存在: {explicit_path}")

    # 2. 标准路径 (与 capacity_pressure_test.py 的 config_dir 约定一致)
    standard_path = os.path.join(output_dir, f"config_{config_name}",
                                 "best_checkpoint.pth")
    if os.path.exists(standard_path):
        return standard_path

    # 3. 不带 config_ 前缀的路径 (兼容手动保存)
    alt_path = os.path.join(output_dir, config_name, "best_checkpoint.pth")
    if os.path.exists(alt_path):
        return alt_path

    # 4. 在标准目录中搜索任意 .pth 文件
    standard_dir = os.path.join(output_dir, f"config_{config_name}")
    if os.path.isdir(standard_dir):
        pth_files = glob.glob(os.path.join(standard_dir, "*.pth"))
        if pth_files:
            # 按修改时间排序, 取最新的
            pth_files.sort(key=os.path.getmtime, reverse=True)
            print(f"  ℹ️  未找到 best_checkpoint.pth, 使用最新的: "
                  f"{os.path.basename(pth_files[0])}")
            return pth_files[0]

    # 5. 在备选目录中搜索
    alt_dir = os.path.join(output_dir, config_name)
    if os.path.isdir(alt_dir):
        pth_files = glob.glob(os.path.join(alt_dir, "*.pth"))
        if pth_files:
            pth_files.sort(key=os.path.getmtime, reverse=True)
            print(f"  ℹ️  未找到 best_checkpoint.pth, 使用最新的: "
                  f"{os.path.basename(pth_files[0])}")
            return pth_files[0]

    return None


# =====================================================================
#  CLI 入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Anla 训练瓶颈诊断",
        epilog="示例:\n"
               "  python -m Anla.diagnostics.diagnostic_probe --config A\n"
               "  python -m Anla.diagnostics.diagnostic_probe --config A "
               "--checkpoint Logs/capacity_pressure_test/config_A_v64_d64/best_checkpoint.pth\n"
               "  python -m Anla.diagnostics.diagnostic_probe --config A --output-dir Logs/capacity_pressure_test\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="A",
        choices=["A", "B", "C", "D", "E"],
        help="配置名 (A/B/C/D/E), 默认 A"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint 文件路径 (默认: 自动在 output-dir 下搜索)"
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(_ANLA_ROOT, "Logs", "capacity_pressure_test"),
        help="输出/日志目录 (默认: Logs/capacity_pressure_test)"
    )
    parser.add_argument(
        "--n-batches", type=int, default=10,
        help="诊断用 batch 数 (默认: 10, 更多更稳定但更慢)"
    )
    args = parser.parse_args()

    # 延迟导入 capacity_pressure_test 的组件
    from Anla.experiments.capacity.capacity_pressure_test import (
        EXPERIMENT_CONFIGS,
        AnlaManifoldInpainter,
        RingSpanDataGeneratorWithHoldout,
    )

    cfg = EXPERIMENT_CONFIGS[args.config]
    device = torch.device("cpu")

    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    mask_id = vocab_size

    # ---- 搜索 checkpoint ----
    ckpt_path = find_checkpoint(
        output_dir=args.output_dir,
        config_name=cfg["name"],
        explicit_path=args.checkpoint,
    )

    # ---- 加载模型 ----
    model = AnlaManifoldInpainter(vocab_size, d_model, num_heads).to(device)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✅ Loaded checkpoint: {ckpt_path}")
        print(f"     Epoch: {ckpt.get('epoch', '?')}, "
              f"Train Acc: {ckpt.get('train_acc', '?')}")
    else:
        # 列出搜索过的路径, 帮助用户排查
        cfg_name = cfg["name"]
        path1 = os.path.join(args.output_dir,
                             f"config_{cfg_name}", "best_checkpoint.pth")
        path2 = os.path.join(args.output_dir,
                             cfg_name, "best_checkpoint.pth")
        print(f"  ❌ 未找到 checkpoint 文件!")
        print(f"     搜索路径:")
        print(f"       1. {path1}")
        print(f"       2. {path2}")
        print(f"     提示: 请确认 capacity_pressure_test 已运行完成,")
        print(f"           或使用 --checkpoint 显式指定 .pth 文件路径。")
        print(f"     当前将使用随机初始化模型运行诊断 (仅作参考)。")
        print()

    # ---- 数据生成器 ----
    gen = RingSpanDataGeneratorWithHoldout(
        vocab_size=vocab_size,
        seq_len=cfg["seq_len"],
        mask_id=mask_id,
        holdout_frac=cfg["holdout_frac"],
        seed=42,
    )

    # ---- 运行诊断 ----
    report = run_diagnostic_with_model(
        model, gen, cfg, device, n_batches=args.n_batches
    )

    # ---- 打印 ----
    print_diagnostic_report(report)

    # ---- [Fix #3] 保存 JSON 到与 checkpoint 同目录 ----
    if ckpt_path is not None:
        # 保存到 checkpoint 所在目录
        report_dir = os.path.dirname(ckpt_path)
    else:
        # 没有 checkpoint 时, 保存到标准目录
        report_dir = os.path.join(args.output_dir, f"config_{cfg['name']}")

    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "diagnostic_report.json")

    # 转换 numpy 类型以便 JSON 序列化
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(report), f, ensure_ascii=False, indent=2)
    print(f"  📄 Report saved to: {out_path}")


if __name__ == "__main__":
    main()
