"""
保存位置: Anla/experiments/nlp/analyze_v9_phase_alignment.py

v9 相位对齐度精细分析
=====================

四个分析模块:

    模块 1: c_m 与 W₁ 列向量的相位对齐度
        核心目标: 区分两种解释
        解释 A (悲观): c_m 的方向学习无效, 相位随机
        解释 B (中性): c_m 与各自的 W₁ 列对齐, 但全局视角下看起来均匀
        方法:
            - 对训练数据做 forward pass, 收集每层 h_m = W₁[m,:] · x_norm 的相位
            - 计算 arg(c_m) 与 E[arg(h_m)] 的相位差 Δφ_m
            - 如果 Δφ 集中在 0 附近 → 解释 B (c 确实在对齐)
            - 如果 Δφ 均匀分布 → 解释 A (c 没有对齐)

    模块 2: Rayleigh R 精确统计量
        对每层的 arg(c) 分布, 计算 Rayleigh 检验的 R 值和 p 值
        R > 0.1 且 p < 0.05 → 分布显著偏离均匀

    模块 3: Attention 层 Im(S) 分布
        验证方案 B (笛卡尔注意力) 的相位旋转是否有结构
        - 计算 Im(Q^H K / √D_h) 的分布
        - 检查是否在 [-π, π] 内有意义地分布 (非退化)

    模块 4: W₂ᶜ 的权重结构分析
        - 检查 W₂ᶜ 的 Re/Im 分量的相关性
        - 分析 W₂ᶜ 的奇异值分布 (与自由实数矩阵对比)

用法:
    cd <项目根目录>
    python -m Anla.experiments.nlp.analyze_v9_phase_alignment \
        --checkpoint Anla/Logs/nlp_byte_mlm_v9/complex/best.pth \
        --output-dir Anla/Logs/nlp_byte_mlm_v9/complex/analysis

    如果 checkpoint 和 data 路径不同, 请自行修改 --checkpoint 和 --data-path 参数。
"""

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- 路径设置 ----
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

# ---- 导入 v9 模型和数据工具 ----
from Anla.experiments.nlp.byte_mlm_v9 import (
    HoloDCUByteMLM, TextByteGenerator, download_tiny_shakespeare,
    DEFAULT_CFG
)

# ---- 全局绘图样式 ----
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#ffffff',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# 颜色方案
LAYER_COLORS = ['#4C72B0', '#DD8452', '#55A868']  # 蓝, 橙, 绿
C_ACCENT = '#C44E52'  # 红
C_GRAY = '#999999'


# =====================================================================
#  工具函数
# =====================================================================

def circular_mean(angles: np.ndarray) -> float:
    """
    计算角度的圆周均值 (circular mean)。

    对角度集合 {θ₁, θ₂, ..., θ_N}, 圆周均值定义为:
        θ̄ = arg(Σ exp(iθ_k))

    Args:
        angles: 角度数组 (弧度), shape (N,)
    Returns:
        圆周均值 (弧度), 范围 [-π, π]
    """
    return np.angle(np.mean(np.exp(1j * angles)))


def rayleigh_test(angles: np.ndarray):
    """
    Rayleigh 均匀性检验。

    检验零假设 H₀: 角度分布为均匀分布 (von Mises with κ=0)。

    统计量:
        R = |Σ exp(iθ_k)| / N    (平均结果长度, 0 ≤ R ≤ 1)
        Z = N · R²                 (Rayleigh 统计量)
        p ≈ exp(-Z) · (1 + (2Z - Z²)/(4N) - (24Z - 132Z² + 76Z³ - 9Z⁴)/(288N²))

    R ≈ 0: 均匀分布
    R ≈ 1: 完全集中在一个方向

    Args:
        angles: 角度数组 (弧度), shape (N,)
    Returns:
        (R, Z, p_value) 三元组
    """
    N = len(angles)
    # 平均结果向量
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = np.sqrt(C**2 + S**2)
    Z = N * R**2

    # Rayleigh 检验的精确 p 值近似 (Mardia & Jupp, 2000)
    if N >= 50:
        # 大样本近似
        p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*N)
                          - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*N**2))
    else:
        # 小样本: 简单指数近似
        p = np.exp(-Z)

    p = max(p, 1e-300)  # 防止下溢
    return R, Z, p


def circular_std(angles: np.ndarray) -> float:
    """
    圆周标准差 (circular standard deviation)。

    定义: σ_c = √(-2 · ln(R)), 其中 R 是平均结果长度。
    当分布集中时 σ_c → 0, 当分布均匀时 σ_c → ∞。

    Args:
        angles: 角度数组 (弧度)
    Returns:
        圆周标准差 (弧度)
    """
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = np.sqrt(C**2 + S**2)
    R = max(R, 1e-10)  # 防止 log(0)
    return np.sqrt(-2.0 * np.log(R))


# =====================================================================
#  模块 1: c_m 与 W₁ 列向量的相位对齐度
# =====================================================================

def analyze_phase_alignment(model, gen, device, num_batches=50):
    """
    分析每个 DCU 通道的本振 c_m 与其接收到的信号 h_m 之间的相位对齐度。

    对每一层 l、每个通道 m:
        1. 收集 h_m = W₁[m,:] · x_norm 在多个 batch 上的相位 arg(h_m)
        2. 计算 arg(h_m) 的圆周均值 θ̄_m = circular_mean(arg(h_m))
        3. 计算相位差 Δφ_m = arg(c_m) - θ̄_m
        4. 检验 Δφ 的分布

    如果 Δφ 集中在 0 附近 (圆周标准差小):
        → 解释 B 成立: c_m 确实与 h_m 的典型相位对齐
        → arg(c) 的全局均匀分布是因为 W₁ 列向量的相位本身均匀
    如果 Δφ 均匀分布:
        → 解释 A 成立: c_m 没有学到有意义的方向

    Args:
        model: 加载了权重的 HoloDCUByteMLM
        gen: TextByteGenerator
        device: torch.device
        num_batches: 采样批次数 (越多统计越稳定)

    Returns:
        dict: 每层的分析结果
    """
    model.eval()
    cfg = DEFAULT_CFG
    num_layers = len(model.blocks)

    # 存储每层每通道的 h_m 相位
    # h_phases[layer_idx] = list of (batch*seq, M) tensors
    h_phases = [[] for _ in range(num_layers)]

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inp, tgt = gen.generate_train_batch(
                cfg['batch_size'], mask_mode='bert', mask_prob=cfg['mask_prob'])
            inp = inp.to(device)

            # 前向传播, 逐层收集中间结果
            z = model.embedding(inp)  # (B, S, D), ℂ

            for layer_idx, block in enumerate(model.blocks):
                # ---- Attention 子层 ----
                h_attn = block.norm1(z)
                h_attn = block.attn(h_attn)
                z = z + h_attn

                # ---- FFN 子层: 在 W₁ 投影后收集 h ----
                x_norm = block.norm2(z)  # (B, S, D), ℂ

                # h = W₁ · x_norm  (与 PhaseFaithfulDCUFFN.forward 的 Step 1 完全一致)
                h = F.linear(x_norm, block.ffn.w1)  # (B, S, M), ℂ

                # 收集 h 的相位: arg(h_m) for each channel m
                # h shape: (B, S, M) → reshape to (B*S, M)
                h_flat = h.reshape(-1, h.shape[-1])  # (B*S, M)
                h_phase = torch.angle(h_flat)  # (B*S, M), ℝ, 范围 [-π, π]
                h_phases[layer_idx].append(h_phase.cpu().numpy())

                # 完成 FFN 子层 (继续前向传播)
                ffn_out = block.ffn(x_norm)
                z = z + ffn_out

    # ---- 逐层分析 ----
    results = {}
    for layer_idx in range(num_layers):
        # 合并所有 batch 的 h 相位
        # all_h_phases shape: (total_tokens, M)
        all_h_phases = np.concatenate(h_phases[layer_idx], axis=0)
        M = all_h_phases.shape[1]

        # 获取本层的 c
        c = model.blocks[layer_idx].ffn.c.detach().cpu()
        c_phase = torch.angle(c).numpy()  # (M,)
        c_mag = c.abs().numpy()             # (M,)

        # 对每个通道 m 计算对齐度
        delta_phases = np.zeros(M)          # Δφ_m = arg(c_m) - circular_mean(arg(h_m))
        h_circular_means = np.zeros(M)      # θ̄_m
        h_rayleigh_Rs = np.zeros(M)         # h_m 相位分布的 Rayleigh R
        alignment_cos = np.zeros(M)         # cos(Δφ_m): 对齐度, 1=完美对齐

        for m in range(M):
            h_m_phases = all_h_phases[:, m]  # (total_tokens,)

            # 圆周均值
            h_mean = circular_mean(h_m_phases)
            h_circular_means[m] = h_mean

            # h_m 相位本身的集中度 (Rayleigh R)
            R_h, _, _ = rayleigh_test(h_m_phases)
            h_rayleigh_Rs[m] = R_h

            # 相位差
            delta = c_phase[m] - h_mean
            # 归一化到 [-π, π]
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            delta_phases[m] = delta

            alignment_cos[m] = np.cos(delta)

        # Δφ 的分布统计
        delta_R, delta_Z, delta_p = rayleigh_test(delta_phases)
        delta_circ_std = circular_std(delta_phases)
        delta_circ_mean = circular_mean(delta_phases)

        # c 相位的 Rayleigh 检验
        c_R, c_Z, c_p = rayleigh_test(c_phase)

        layer_result = {
            'layer': layer_idx,
            'M': M,
            'num_tokens': all_h_phases.shape[0],

            # c 自身的相位统计
            'c_rayleigh_R': float(c_R),
            'c_rayleigh_Z': float(c_Z),
            'c_rayleigh_p': float(c_p),

            # h_m 相位集中度 (每通道的 Rayleigh R)
            'h_rayleigh_R_mean': float(np.mean(h_rayleigh_Rs)),
            'h_rayleigh_R_std': float(np.std(h_rayleigh_Rs)),
            'h_rayleigh_R_min': float(np.min(h_rayleigh_Rs)),
            'h_rayleigh_R_max': float(np.max(h_rayleigh_Rs)),

            # Δφ 分布统计 (核心指标)
            'delta_rayleigh_R': float(delta_R),
            'delta_rayleigh_Z': float(delta_Z),
            'delta_rayleigh_p': float(delta_p),
            'delta_circular_mean': float(delta_circ_mean),
            'delta_circular_std': float(delta_circ_std),

            # 对齐度统计
            'alignment_cos_mean': float(np.mean(alignment_cos)),
            'alignment_cos_std': float(np.std(alignment_cos)),
            'alignment_cos_median': float(np.median(alignment_cos)),
            'frac_aligned': float(np.mean(alignment_cos > 0)),
            # cos(Δφ) > 0 表示 Δφ ∈ (-π/2, π/2), 即大致对齐
            # 如果均匀分布, frac_aligned ≈ 0.5
            # 如果完美对齐, frac_aligned = 1.0

            # 原始数据 (用于绘图)
            'delta_phases': delta_phases.tolist(),
            'c_phase': c_phase.tolist(),
            'c_mag': c_mag.tolist(),
            'h_circular_means': h_circular_means.tolist(),
            'h_rayleigh_Rs': h_rayleigh_Rs.tolist(),
            'alignment_cos': alignment_cos.tolist(),
        }
        results[f'layer_{layer_idx}'] = layer_result

    return results


# =====================================================================
#  模块 3: Attention 层 Im(S) 分布分析
# =====================================================================

def analyze_attention_phases(model, gen, device, num_batches=20):
    """
    分析笛卡尔正交分解注意力中 Im(S) 的分布。

    对每一层、每个头:
        1. 计算 S = Q^H K / √D_h (复数 score)
        2. 提取 Re(S) 和 Im(S) 的分布统计
        3. 检查 Im(S) 是否在有意义的范围内 (非退化)

    如果 Im(S) ≈ 0 (退化):
        → 方案 B 等价于标准实数 attention, 相位旋转不起作用
    如果 Im(S) 的标准差 > 0.5:
        → 方案 B 的相位旋转是活跃的, exp(i·Im(S)) 产生了有意义的旋转

    Args:
        model: 加载了权重的 HoloDCUByteMLM
        gen: TextByteGenerator
        device: torch.device
        num_batches: 采样批次数

    Returns:
        dict: 每层每头的分析结果
    """
    model.eval()
    cfg = DEFAULT_CFG
    num_layers = len(model.blocks)

    # 存储 score 的 Re 和 Im
    # scores_re[layer][head] = list of arrays
    scores_re = [[[] for _ in range(model.blocks[0].attn.num_heads)]
                 for _ in range(num_layers)]
    scores_im = [[[] for _ in range(model.blocks[0].attn.num_heads)]
                 for _ in range(num_layers)]

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inp, tgt = gen.generate_train_batch(
                cfg['batch_size'], mask_mode='bert', mask_prob=cfg['mask_prob'])
            inp = inp.to(device)

            z = model.embedding(inp)  # (B, S, D)

            for layer_idx, block in enumerate(model.blocks):
                # ---- 手动展开 attention forward, 收集 score ----
                x_norm = block.norm1(z)
                B, S, D = x_norm.shape
                H = block.attn.num_heads
                Dh = block.attn.d_head

                Q = F.linear(x_norm, block.attn.wq).view(B, S, H, Dh).transpose(1, 2)
                K = F.linear(x_norm, block.attn.wk).view(B, S, H, Dh).transpose(1, 2)
                V = F.linear(x_norm, block.attn.wv).view(B, S, H, Dh).transpose(1, 2)

                Q = block.attn.rotary(Q)
                K = block.attn.rotary(K)

                # Hermitian 内积
                score = torch.matmul(Q, K.transpose(-2, -1).conj())  # (B, H, S, S)
                scale = 1.0 / math.sqrt(Dh)
                score = score * scale

                # 收集 Re(S) 和 Im(S)
                for h in range(H):
                    s_h = score[:, h, :, :]  # (B, S, S)
                    scores_re[layer_idx][h].append(s_h.real.cpu().numpy().ravel())
                    scores_im[layer_idx][h].append(s_h.imag.cpu().numpy().ravel())

                # 完成 attention + FFN (继续前向传播)
                attn_out = block.attn(block.norm1(z))
                z = z + attn_out
                ffn_out = block.ffn(block.norm2(z))
                z = z + ffn_out

    # ---- 汇总统计 ----
    results = {}
    for layer_idx in range(num_layers):
        H = len(scores_re[layer_idx])
        layer_result = {'layer': layer_idx, 'heads': {}}

        for h in range(H):
            re_all = np.concatenate(scores_re[layer_idx][h])
            im_all = np.concatenate(scores_im[layer_idx][h])

            # Im(S) 对应 exp(i·Im(S)) 的旋转角度
            # 有效旋转范围: 如果 |Im(S)| > π, 旋转超过半周
            phase_rotations = im_all  # 这就是 exp(i·Im(S)) 的角度

            head_result = {
                # Re(S) 统计 (softmax 输入)
                're_mean': float(np.mean(re_all)),
                're_std': float(np.std(re_all)),
                're_min': float(np.percentile(re_all, 1)),
                're_max': float(np.percentile(re_all, 99)),
                # 注: Re(S) 可以为负 → 支持"强不关注"

                # Im(S) 统计 (相位旋转角度)
                'im_mean': float(np.mean(im_all)),
                'im_std': float(np.std(im_all)),
                'im_min': float(np.percentile(im_all, 1)),
                'im_max': float(np.percentile(im_all, 99)),

                # 有效旋转分析
                'frac_rotation_gt_pi4': float(np.mean(np.abs(im_all) > np.pi/4)),
                'frac_rotation_gt_pi2': float(np.mean(np.abs(im_all) > np.pi/2)),
                'frac_rotation_gt_pi': float(np.mean(np.abs(im_all) > np.pi)),
                # frac > π/4: 超过 45° 旋转的比例
                # frac > π/2: 超过 90° 旋转的比例
                # frac > π:   超过 180° 旋转 (回绕) 的比例

                # Re(S) 负值比例 (v8 MagPhaseSoftmax 中 |S| ≥ 0, 没有负值)
                'frac_re_negative': float(np.mean(re_all < 0)),
            }
            layer_result['heads'][f'head_{h}'] = head_result

        results[f'layer_{layer_idx}'] = layer_result

    return results


# =====================================================================
#  模块 4: W₂ᶜ 权重结构分析
# =====================================================================

def analyze_w2_structure(model):
    """
    分析 W₂ᶜ ∈ ℂ^{D×M} 的内部结构。

    W₂ᶜ 在实数展开下等价于 ⎡ A  -B ⎤ (A = Re(W₂), B = Im(W₂))
                              ⎣ B   A ⎦
    检查:
        1. Re(W₂) 和 Im(W₂) 的统计特征 (均值/方差/范围)
        2. Re(W₂) 和 Im(W₂) 之间的相关性
           (如果高度相关 → 矩阵接近实数矩阵乘以 e^{iφ};
            如果不相关 → 充分利用了复数结构)
        3. W₂ᶜ 的奇异值分布 (有效秩)

    Returns:
        dict: 每层的 W₂ 结构分析
    """
    results = {}
    for layer_idx, block in enumerate(model.blocks):
        w2 = block.ffn.w2.detach().cpu()  # ℂ^{D×M}
        D, M = w2.shape

        w2_re = w2.real.numpy()  # (D, M)
        w2_im = w2.imag.numpy()  # (D, M)

        # ---- 基本统计 ----
        re_stats = {
            'mean': float(np.mean(w2_re)),
            'std': float(np.std(w2_re)),
            'abs_mean': float(np.mean(np.abs(w2_re))),
        }
        im_stats = {
            'mean': float(np.mean(w2_im)),
            'std': float(np.std(w2_im)),
            'abs_mean': float(np.mean(np.abs(w2_im))),
        }

        # ---- Re-Im 相关性 ----
        # 将 Re 和 Im 展平, 计算 Pearson 相关系数
        corr = np.corrcoef(w2_re.ravel(), w2_im.ravel())[0, 1]

        # 逐行 (逐输出维度) 的 Re-Im 相关
        row_corrs = []
        for d in range(D):
            if np.std(w2_re[d]) > 1e-10 and np.std(w2_im[d]) > 1e-10:
                row_corrs.append(np.corrcoef(w2_re[d], w2_im[d])[0, 1])
        row_corrs = np.array(row_corrs)

        # ---- 奇异值分析 ----
        # 复数 SVD
        U, S_vals, Vh = np.linalg.svd(w2.numpy(), full_matrices=False)
        S_vals_real = np.abs(S_vals)  # 奇异值 (已经是实数非负)
        # 有效秩: exp(H(p)), 其中 p_k = σ_k / Σσ
        p = S_vals_real / S_vals_real.sum()
        p = p[p > 1e-10]  # 去掉零值
        effective_rank = np.exp(-np.sum(p * np.log(p)))
        # 归一化有效秩 (0-1)
        max_rank = min(D, M)
        norm_effective_rank = effective_rank / max_rank

        # ---- W₂ᶜ 的相位结构 ----
        # 每个元素 W₂[d,m] 的相位
        w2_phase = np.angle(w2.numpy())
        # 如果 W₂ 接近实数矩阵 (Im ≈ 0), 相位集中在 0 或 π
        # 如果充分利用复数, 相位均匀分布
        w2_phase_flat = w2_phase.ravel()
        phase_R, _, phase_p = rayleigh_test(w2_phase_flat)

        layer_result = {
            'layer': layer_idx,
            'shape': [D, M],
            're_stats': re_stats,
            'im_stats': im_stats,
            'global_re_im_corr': float(corr),
            'row_re_im_corr_mean': float(np.mean(row_corrs)) if len(row_corrs) > 0 else 0.0,
            'row_re_im_corr_std': float(np.std(row_corrs)) if len(row_corrs) > 0 else 0.0,
            'singular_values': S_vals_real.tolist(),
            'effective_rank': float(effective_rank),
            'norm_effective_rank': float(norm_effective_rank),
            'max_rank': max_rank,
            'phase_rayleigh_R': float(phase_R),
            'phase_rayleigh_p': float(phase_p),
            're_im_ratio': float(np.std(w2_re) / max(np.std(w2_im), 1e-10)),
            # re_im_ratio ≈ 1: Re 和 Im 被同等利用
            # re_im_ratio >> 1: Im 几乎不用 (退化为实数矩阵)
        }
        results[f'layer_{layer_idx}'] = layer_result

    return results


# =====================================================================
#  可视化
# =====================================================================

def plot_phase_alignment(alignment_results, output_dir):
    """
    绘制相位对齐度分析的可视化 (4 行 × 3 列 = 12 个面板)。
    """
    num_layers = len(alignment_results)

    # ---- 图 A: Δφ 分布 (核心) ----
    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 9))
    fig.suptitle('Module 1: Phase Alignment — Δφ = arg(c_m) − E[arg(h_m)]\n'
                 'If concentrated near 0 → Explanation B (c aligns with h)\n'
                 'If uniform → Explanation A (c is random)',
                 fontsize=12, fontweight='bold', y=1.02)

    for i in range(num_layers):
        res = alignment_results[f'layer_{i}']
        delta = np.array(res['delta_phases'])
        cos_align = np.array(res['alignment_cos'])

        # 上行: Δφ 直方图
        ax = axes[0, i]
        ax.hist(delta, bins=36, range=(-np.pi, np.pi), density=True,
                color=LAYER_COLORS[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        # 均匀分布基线
        ax.axhline(y=1/(2*np.pi), color=C_GRAY, linestyle='--', linewidth=1.5,
                    label=f'Uniform (1/2π = {1/(2*np.pi):.3f})')
        # 标注统计量
        ax.text(0.02, 0.98,
                f'Rayleigh R = {res["delta_rayleigh_R"]:.4f}\n'
                f'p = {res["delta_rayleigh_p"]:.2e}\n'
                f'circ_std = {res["delta_circular_std"]:.3f}\n'
                f'circ_mean = {res["delta_circular_mean"]*180/np.pi:.1f}°',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Δφ (rad)')
        ax.set_ylabel('Density')
        ax.set_title(f'Layer {i}: Δφ Distribution')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim(-np.pi, np.pi)

        # 下行: cos(Δφ) 直方图
        ax = axes[1, i]
        ax.hist(cos_align, bins=40, range=(-1, 1), density=True,
                color=LAYER_COLORS[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(x=np.mean(cos_align), color='red', linestyle='--', linewidth=2,
                   label=f'mean = {np.mean(cos_align):.4f}')
        frac = res['frac_aligned']
        ax.text(0.02, 0.98,
                f'E[cos(Δφ)] = {np.mean(cos_align):.4f}\n'
                f'frac(cos>0) = {frac:.3f}\n'
                f'(random=0.500)',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('cos(Δφ)')
        ax.set_ylabel('Density')
        ax.set_title(f'Layer {i}: Alignment cos(Δφ)')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'phase_alignment_delta.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] phase_alignment_delta.png')

    # ---- 图 B: h_m 信号本身的相位集中度 ----
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4.5))
    fig.suptitle('Module 1 (aux): Signal Phase Concentration — Rayleigh R of arg(h_m)',
                 fontsize=12, fontweight='bold', y=1.03)

    for i in range(num_layers):
        res = alignment_results[f'layer_{i}']
        h_Rs = np.array(res['h_rayleigh_Rs'])
        ax = axes[i] if num_layers > 1 else axes
        ax.hist(h_Rs, bins=40, color=LAYER_COLORS[i], alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=np.mean(h_Rs), color='red', linestyle='--', linewidth=2,
                   label=f'mean R = {np.mean(h_Rs):.4f}')
        ax.text(0.98, 0.98,
                f'R mean = {res["h_rayleigh_R_mean"]:.4f}\n'
                f'R std = {res["h_rayleigh_R_std"]:.4f}\n'
                f'R range = [{res["h_rayleigh_R_min"]:.4f},\n'
                f'           {res["h_rayleigh_R_max"]:.4f}]',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Rayleigh R of arg(h_m)')
        ax.set_ylabel('Count')
        ax.set_title(f'Layer {i}: Signal Phase Concentration')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'phase_alignment_h_concentration.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] phase_alignment_h_concentration.png')

    # ---- 图 C: c_m 相位 vs h_m 圆周均值相位 (散点图) ----
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4.5))
    fig.suptitle('Module 1: arg(c_m) vs circular_mean(arg(h_m))\n'
                 'If aligned → points cluster near diagonal',
                 fontsize=12, fontweight='bold', y=1.05)

    for i in range(num_layers):
        res = alignment_results[f'layer_{i}']
        c_ph = np.array(res['c_phase'])
        h_mean = np.array(res['h_circular_means'])
        cos_al = np.array(res['alignment_cos'])

        ax = axes[i] if num_layers > 1 else axes
        sc = ax.scatter(h_mean * 180/np.pi, c_ph * 180/np.pi,
                        c=cos_al, cmap='RdYlGn', vmin=-1, vmax=1,
                        s=10, alpha=0.6, edgecolors='none')
        ax.plot([-180, 180], [-180, 180], 'k--', linewidth=1, alpha=0.5,
                label='Perfect alignment')
        ax.set_xlabel('circular_mean(arg(h_m)) (°)')
        ax.set_ylabel('arg(c_m) (°)')
        ax.set_title(f'Layer {i}')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='upper left')
        fig.colorbar(sc, ax=ax, label='cos(Δφ)', shrink=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'phase_alignment_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] phase_alignment_scatter.png')


def plot_attention_analysis(attn_results, output_dir):
    """
    绘制 attention 层 Im(S) 分布的可视化。
    """
    num_layers = len(attn_results)
    # 假设每层头数相同
    first_layer = attn_results[f'layer_0']
    num_heads = len(first_layer['heads'])

    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4*num_layers))
    fig.suptitle('Module 3: Cartesian Attention — Score Decomposition\n'
                 'Re(S) → softmax selection | Im(S) → phase rotation exp(i·Im(S))',
                 fontsize=12, fontweight='bold', y=1.02)

    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_layers):
        res = attn_results[f'layer_{i}']

        # 汇总所有头的统计
        re_means = [res['heads'][f'head_{h}']['re_mean'] for h in range(num_heads)]
        re_stds = [res['heads'][f'head_{h}']['re_std'] for h in range(num_heads)]
        im_means = [res['heads'][f'head_{h}']['im_mean'] for h in range(num_heads)]
        im_stds = [res['heads'][f'head_{h}']['im_std'] for h in range(num_heads)]
        frac_neg = [res['heads'][f'head_{h}']['frac_re_negative'] for h in range(num_heads)]
        frac_gt_pi4 = [res['heads'][f'head_{h}']['frac_rotation_gt_pi4'] for h in range(num_heads)]
        frac_gt_pi2 = [res['heads'][f'head_{h}']['frac_rotation_gt_pi2'] for h in range(num_heads)]

        # 左图: Re(S) 和 Im(S) 的 std 对比 (每头)
        ax = axes[i, 0]
        x = np.arange(num_heads)
        width = 0.35
        ax.bar(x - width/2, re_stds, width, color='#4C72B0', alpha=0.8,
               label='std(Re(S))')
        ax.bar(x + width/2, im_stds, width, color='#DD8452', alpha=0.8,
               label='std(Im(S))')
        for h in range(num_heads):
            ax.text(h - width/2, re_stds[h] + 0.02, f'{re_stds[h]:.2f}',
                    ha='center', fontsize=7)
            ax.text(h + width/2, im_stds[h] + 0.02, f'{im_stds[h]:.2f}',
                    ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'H{h}' for h in range(num_heads)])
        ax.set_ylabel('Std Dev')
        ax.set_title(f'Layer {i}: Score Component Spread')
        ax.legend()

        # 右图: 相位旋转活跃度
        ax = axes[i, 1]
        x = np.arange(num_heads)
        width = 0.25
        ax.bar(x - width, frac_neg, width, color='#C44E52', alpha=0.8,
               label='Re(S)<0 (anti-attend)')
        ax.bar(x, frac_gt_pi4, width, color='#55A868', alpha=0.8,
               label='|Im(S)|>π/4 (45° rot)')
        ax.bar(x + width, frac_gt_pi2, width, color='#8172B2', alpha=0.8,
               label='|Im(S)|>π/2 (90° rot)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'H{h}' for h in range(num_heads)])
        ax.set_ylabel('Fraction')
        ax.set_title(f'Layer {i}: Attention Feature Activity')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attention_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] attention_analysis.png')


def plot_w2_analysis(w2_results, output_dir):
    """
    绘制 W₂ᶜ 权重结构分析的可视化。
    """
    num_layers = len(w2_results)

    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 8))
    fig.suptitle('Module 4: W₂ᶜ Weight Structure Analysis\n'
                 'How does the U(1)-constrained complex matrix utilize its structure?',
                 fontsize=12, fontweight='bold', y=1.02)

    if num_layers == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_layers):
        res = w2_results[f'layer_{i}']

        # 上行: 奇异值分布
        ax = axes[0, i]
        sv = np.array(res['singular_values'])
        ax.bar(range(len(sv)), sv, color=LAYER_COLORS[i], alpha=0.7,
               edgecolor='black', linewidth=0.3)
        ax.axhline(y=sv.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'mean = {sv.mean():.4f}')
        ax.text(0.98, 0.98,
                f'eff. rank = {res["effective_rank"]:.1f}\n'
                f'(norm = {res["norm_effective_rank"]:.3f})\n'
                f'max rank = {res["max_rank"]}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('σ_k')
        ax.set_title(f'Layer {i}: W₂ᶜ Singular Values')
        ax.legend(fontsize=7)

        # 下行: Re-Im 利用度
        ax = axes[1, i]
        labels = ['Re/Im\nStd Ratio', 'Global\nRe-Im Corr', 'Row Mean\nRe-Im Corr']
        values = [res['re_im_ratio'], abs(res['global_re_im_corr']),
                  abs(res['row_re_im_corr_mean'])]
        colors_bar = ['#4C72B0', '#DD8452', '#55A868']
        bars = ax.bar(labels, values, color=colors_bar, alpha=0.8,
                      edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1,
                   label='Re=Im balance')
        ax.set_ylabel('Value')
        ax.set_title(f'Layer {i}: Re-Im Structure')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'w2_structure_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] w2_structure_analysis.png')


# =====================================================================
#  汇总报告
# =====================================================================

def print_summary(alignment_results, attn_results, w2_results):
    """
    打印人类可读的汇总诊断报告。
    """
    print()
    print('=' * 80)
    print('                   v9 相位对齐度精细分析 — 汇总报告')
    print('=' * 80)

    # ---- 模块 1: 相位对齐 ----
    print()
    print('━' * 80)
    print('  模块 1: c_m 与 W₁ 信号的相位对齐度')
    print('━' * 80)
    print()
    print(f'  {"Layer":>6s}  {"Δφ R":>8s}  {"Δφ p":>12s}  {"E[cos(Δφ)]":>12s}  '
          f'{"frac(cos>0)":>12s}  {"circ_std":>10s}  {"h_R_mean":>10s}')
    print(f'  {"":-<6s}  {"":-<8s}  {"":-<12s}  {"":-<12s}  {"":-<12s}  {"":-<10s}  {"":-<10s}')

    for key in sorted(alignment_results.keys()):
        res = alignment_results[key]
        # 判定
        if res['delta_rayleigh_p'] < 0.01 and res['alignment_cos_mean'] > 0.05:
            verdict = '✓ ALIGNED'
        elif res['delta_rayleigh_p'] < 0.05:
            verdict = '~ WEAK'
        else:
            verdict = '✗ RANDOM'

        print(f'  {res["layer"]:>6d}  {res["delta_rayleigh_R"]:>8.4f}  '
              f'{res["delta_rayleigh_p"]:>12.2e}  {res["alignment_cos_mean"]:>12.4f}  '
              f'{res["frac_aligned"]:>12.3f}  {res["delta_circular_std"]:>10.3f}  '
              f'{res["h_rayleigh_R_mean"]:>10.4f}  {verdict}')

    print()
    print('  解读:')
    print('    Δφ R > 0.1 且 p < 0.05 → 相位差显著偏离均匀 → 解释 B (c 在对齐)')
    print('    E[cos(Δφ)] > 0 → 对齐方向正确 (Δφ 偏向 0 而非 π)')
    print('    frac(cos>0) > 0.5 → 多数通道大致对齐')
    print('    h_R_mean ≈ 0 → h_m 的相位本身就不集中 (输入依赖性强)')

    # ---- 模块 2: c 相位 Rayleigh 检验 ----
    print()
    print('━' * 80)
    print('  模块 2: 本振相位 arg(c) 的 Rayleigh 检验')
    print('━' * 80)
    print()
    for key in sorted(alignment_results.keys()):
        res = alignment_results[key]
        sig = '***' if res['c_rayleigh_p'] < 0.001 else \
              '**' if res['c_rayleigh_p'] < 0.01 else \
              '*' if res['c_rayleigh_p'] < 0.05 else 'ns'
        print(f'  Layer {res["layer"]}: R = {res["c_rayleigh_R"]:.4f}, '
              f'Z = {res["c_rayleigh_Z"]:.2f}, p = {res["c_rayleigh_p"]:.4e} {sig}')

    # ---- 模块 3: Attention ----
    print()
    print('━' * 80)
    print('  模块 3: Cartesian Attention — Im(S) 活跃度')
    print('━' * 80)
    print()
    for key in sorted(attn_results.keys()):
        res = attn_results[key]
        print(f'  Layer {res["layer"]}:')
        for hkey in sorted(res['heads'].keys()):
            h = res['heads'][hkey]
            print(f'    {hkey}: Re(S) std={h["re_std"]:.3f}, Im(S) std={h["im_std"]:.3f}, '
                  f'Re<0: {h["frac_re_negative"]:.1%}, '
                  f'|Im|>π/4: {h["frac_rotation_gt_pi4"]:.1%}, '
                  f'|Im|>π/2: {h["frac_rotation_gt_pi2"]:.1%}')
    print()
    print('  解读:')
    print('    Im(S) std > 0.5 → 相位旋转活跃, 方案 B 在起作用')
    print('    Im(S) std ≈ 0 → 相位旋转退化, 等价于实数 attention')
    print('    Re(S)<0 比例 → 负 score 提供"强不关注"能力 (v8 MagPhaseSoftmax 无此特性)')

    # ---- 模块 4: W₂ ----
    print()
    print('━' * 80)
    print('  模块 4: W₂ᶜ 权重结构')
    print('━' * 80)
    print()
    for key in sorted(w2_results.keys()):
        res = w2_results[key]
        print(f'  Layer {res["layer"]}: '
              f'Re/Im ratio={res["re_im_ratio"]:.3f}, '
              f'global corr={res["global_re_im_corr"]:.4f}, '
              f'eff. rank={res["effective_rank"]:.1f}/{res["max_rank"]}, '
              f'phase R={res["phase_rayleigh_R"]:.4f}')
    print()
    print('  解读:')
    print('    Re/Im ratio ≈ 1.0 → Re 和 Im 被同等利用 (充分使用复数结构)')
    print('    Re/Im ratio >> 1 → Im 退化, 接近实数矩阵')
    print('    global corr ≈ 0 → Re 和 Im 独立 (正常)')
    print('    global corr ≈ ±1 → Re ≈ ±Im, 矩阵退化为 rank-1 复数结构')


# =====================================================================
#  主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='v9 Phase Alignment Fine-Grained Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='v9 complex model 的 best.pth 路径')
    parser.add_argument('--data-path', type=str, default=None,
                        help='数据文件路径 (默认自动下载)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='分析结果输出目录')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-batches', type=int, default=50,
                        help='模块 1 和 3 的采样批次数 (默认 50)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.checkpoint), 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Output: {args.output_dir}')

    # ---- 加载数据 ----
    if args.data_path is None:
        data_path = download_tiny_shakespeare(
            os.path.join(_ANLA_ROOT, 'data'))
    else:
        data_path = args.data_path

    cfg = DEFAULT_CFG
    gen = TextByteGenerator(
        data_path, cfg['seq_len'], mask_id=256, test_frac=0.1)

    # ---- 加载模型 ----
    print(f'\n  加载模型: {args.checkpoint}')
    model = HoloDCUByteMLM(
        vocab_size=256,
        d_model=cfg['d_model_complex'],
        num_heads=cfg['num_heads'],
        num_blocks=cfg['num_blocks'],
        ff_mult=cfg['ff_mult'],
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    print(f'  参数量: {n_params:,} (实数等效: {n_real:,})')

    # ---- 模块 1 & 2: 相位对齐度分析 ----
    print(f'\n{"="*60}')
    print(f'  模块 1 & 2: 相位对齐度分析 (num_batches={args.num_batches})')
    print(f'{"="*60}')
    alignment_results = analyze_phase_alignment(
        model, gen, device, num_batches=args.num_batches)
    print('  计算完成。')

    # ---- 模块 3: Attention 分析 ----
    print(f'\n{"="*60}')
    print(f'  模块 3: Attention 相位旋转分析')
    print(f'{"="*60}')
    attn_results = analyze_attention_phases(
        model, gen, device, num_batches=min(args.num_batches, 20))
    print('  计算完成。')

    # ---- 模块 4: W₂ 结构分析 ----
    print(f'\n{"="*60}')
    print(f'  模块 4: W₂ᶜ 权重结构分析')
    print(f'{"="*60}')
    w2_results = analyze_w2_structure(model)
    print('  计算完成。')

    # ---- 汇总报告 ----
    print_summary(alignment_results, attn_results, w2_results)

    # ---- 保存 JSON ----
    # 从 alignment_results 中移除大数组 (绘图后不需要保存全部原始数据)
    json_results = {
        'alignment': {},
        'attention': attn_results,
        'w2_structure': w2_results,
    }
    for key, res in alignment_results.items():
        # 复制除大数组外的所有字段
        json_res = {k: v for k, v in res.items()
                    if k not in ('delta_phases', 'c_phase', 'c_mag',
                                 'h_circular_means', 'h_rayleigh_Rs',
                                 'alignment_cos')}
        json_results['alignment'][key] = json_res

    json_path = os.path.join(args.output_dir, 'analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f'\n  [Saved] {json_path}')

    # ---- 绘图 ----
    print(f'\n  绘图中...')
    plot_phase_alignment(alignment_results, args.output_dir)
    plot_attention_analysis(attn_results, args.output_dir)
    plot_w2_analysis(w2_results, args.output_dir)

    print(f'\n{"="*60}')
    print(f'  全部分析完成。结果保存在: {args.output_dir}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
