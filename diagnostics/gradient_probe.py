"""
保存位置: Anla/diagnostics/gradient_probe.py

梯度探针 — 诊断力向量在 manual_backward 链路中的衰减
=========================================================================

目的:
    在训练过程中, 力向量沿 block → rotary → embedding 链路反传。
    如果到达 embedding 的信号过弱或方向混乱, embedding 无法学到结构。

    本脚本在不改变任何训练行为的前提下, 记录力向量在每一层的:
      - L2 范数 (衰减程度)
      - 逐元素标准差 (信号离散度)
      - 最大元素模长 (峰值信号)
      - 方向余弦相似度 (相邻层之间的方向保持度)
      - embedding 实际权重更新量

    同时对比不同 vocab/d_model 配置下的衰减模式。

用法:
    python -m Anla.diagnostics.gradient_probe
    python -m Anla.diagnostics.gradient_probe --configs A D E
    python -m Anla.diagnostics.gradient_probe --epochs 500 --probe-interval 50
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

# [Path Fix] 文件位置: Anla/diagnostics/gradient_probe.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
#  实验配置 (与 capacity_pressure_test.py 一致)
# =====================================================================
PROBE_CONFIGS = {
    "A": {
        "name": "A_v64_d64",
        "vocab_size": 64,
        "d_model": 64,
        "num_heads": 4,
        "seq_len": 32,
        "max_span_length": 5,
        "batch_size": 16,
        "lr": 0.001,
        "weight_decay": 1e-4,
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
        "weight_decay": 1e-4,
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
        "weight_decay": 1e-4,
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
        "weight_decay": 1e-4,
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
        "weight_decay": 1e-4,
    },
}


# =====================================================================
#  探针模型: 拆解 manual_backward, 暴露每层的力向量
# =====================================================================
class ProbedInpainter(nn.Module):
    """
    与 AnlaManifoldInpainter 完全相同的架构,
    但 manual_backward 被拆解为逐层调用, 每层之间插入探针。
    """

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

    def probed_backward(self, force: torch.Tensor, lr: float, wd: float) -> Dict[str, Any]:
        """
        逐层反传, 在每层之间记录力向量的统计量。

        Returns
        -------
        probe_data : dict
            包含每层的力向量诊断数据
        """
        probe = {}

        # ---- 探测点 0: L_Elegant 输出的原始力向量 ----
        probe["force_input"] = _measure_complex_tensor(force, "force_input")

        # ---- 记录 embedding 更新前的权重快照 ----
        emb_before = self.embedding.weight.data.clone()

        # ---- Layer 1: block.manual_backward ----
        grad_after_block = self.block.manual_backward(force, lr, wd)
        probe["after_block"] = _measure_complex_tensor(grad_after_block, "after_block")

        # ---- Layer 2: rotary.manual_backward ----
        grad_after_rotary = self.rotary.manual_backward(grad_after_block)
        probe["after_rotary"] = _measure_complex_tensor(grad_after_rotary, "after_rotary")

        # ---- 方向保持度: force_input vs after_block ----
        probe["direction_block"] = _direction_similarity(force, grad_after_block)

        # ---- 方向保持度: after_block vs after_rotary ----
        probe["direction_rotary"] = _direction_similarity(grad_after_block, grad_after_rotary)

        # ---- Layer 3: embedding.manual_backward ----
        self.embedding.manual_backward(grad_after_rotary, lr, wd)

        # ---- 探测点 3: embedding 实际权重变化量 ----
        emb_after = self.embedding.weight.data
        emb_delta = emb_after - emb_before
        probe["embedding_delta"] = _measure_complex_tensor(emb_delta, "embedding_delta")

        # ---- 额外: embedding 权重本身的统计 ----
        probe["embedding_weight"] = _measure_complex_tensor(
            self.embedding.weight.data, "embedding_weight")

        # ---- 衰减比 ----
        input_norm = probe["force_input"]["l2_norm"]
        after_block_norm = probe["after_block"]["l2_norm"]
        after_rotary_norm = probe["after_rotary"]["l2_norm"]
        delta_norm = probe["embedding_delta"]["l2_norm"]

        probe["attenuation"] = {
            "block_ratio": after_block_norm / (input_norm + 1e-15),
            "rotary_ratio": after_rotary_norm / (after_block_norm + 1e-15),
            "total_ratio": after_rotary_norm / (input_norm + 1e-15),
            "delta_vs_input": delta_norm / (input_norm + 1e-15),
            "delta_vs_weight": delta_norm / (probe["embedding_weight"]["l2_norm"] + 1e-15),
        }

        return probe


# =====================================================================
#  复数张量测量工具
# =====================================================================
def _measure_complex_tensor(t: torch.Tensor, name: str) -> Dict[str, float]:
    """测量复数张量的关键统计量。"""
    with torch.no_grad():
        # 转为实虚拼接以计算范数
        if torch.is_complex(t):
            flat_real = torch.cat([t.real.reshape(-1), t.imag.reshape(-1)])
            mags = t.abs().reshape(-1)
        else:
            flat_real = t.reshape(-1).float()
            mags = t.abs().reshape(-1).float()

        l2 = float(flat_real.norm(p=2))
        l_inf = float(mags.max()) if mags.numel() > 0 else 0.0
        mean_mag = float(mags.mean()) if mags.numel() > 0 else 0.0
        std_mag = float(mags.std()) if mags.numel() > 1 else 0.0

        # 非零元素比例 (判断稀疏性)
        nonzero_frac = float((mags > 1e-10).float().mean()) if mags.numel() > 0 else 0.0

    return {
        "name": name,
        "l2_norm": l2,
        "l_inf": l_inf,
        "mean_magnitude": mean_mag,
        "std_magnitude": std_mag,
        "nonzero_fraction": nonzero_frac,
        "numel": int(mags.numel()),
    }


def _direction_similarity(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """
    计算两个复数张量之间的方向相似度。
    使用实虚拼接后的余弦相似度。
    """
    with torch.no_grad():
        if torch.is_complex(a):
            fa = torch.cat([a.real.reshape(-1), a.imag.reshape(-1)]).float()
        else:
            fa = a.reshape(-1).float()

        if torch.is_complex(b):
            fb = torch.cat([b.real.reshape(-1), b.imag.reshape(-1)]).float()
        else:
            fb = b.reshape(-1).float()

        # 处理形状不匹配 (不同层输出维度可能不同)
        min_len = min(fa.shape[0], fb.shape[0])
        if min_len == 0:
            return {"cosine_similarity": 0.0, "shape_matched": False}

        fa = fa[:min_len]
        fb = fb[:min_len]

        norm_a = fa.norm(p=2)
        norm_b = fb.norm(p=2)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return {"cosine_similarity": 0.0, "shape_matched": True}

        cos_sim = float(torch.dot(fa, fb) / (norm_a * norm_b))

    return {
        "cosine_similarity": cos_sim,
        "shape_matched": fa.shape[0] == fb.shape[0],
    }


# =====================================================================
#  L_Elegant (与训练脚本完全一致)
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
#  数据生成器 (简化版, 不需要 holdout)
# =====================================================================
class RingSpanDataGenerator:
    def __init__(self, vocab_size, seq_len, mask_id):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id

    def generate_batch(self, batch_size, max_span=5):
        inputs, targets = [], []
        for _ in range(batch_size):
            start = random.randint(0, self.vocab_size - 1)
            seq = [(start + i) % self.vocab_size for i in range(self.seq_len)]
            seq_t = torch.tensor(seq, dtype=torch.long)
            inp = seq_t.clone()
            tgt = torch.full_like(seq_t, -100)
            ml = random.randint(1, max_span)
            ms = random.randint(0, self.seq_len - ml)
            inp[ms:ms+ml] = self.mask_id
            tgt[ms:ms+ml] = seq_t[ms:ms+ml]
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


# =====================================================================
#  单配置探测
# =====================================================================
def probe_single_config(
    cfg: Dict[str, Any],
    epochs: int,
    probe_interval: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    对单个配置运行带探针的训练, 返回完整的梯度诊断数据。
    """
    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    seq_len = cfg["seq_len"]
    max_span = cfg["max_span_length"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    mask_id = vocab_size

    print(f"\n{'='*64}")
    print(f"  Probing: {cfg['name']}  "
          f"(V={vocab_size}, D={d_model}, ratio={vocab_size/d_model:.0f}:1)")
    print(f"{'='*64}")

    gen = RingSpanDataGenerator(vocab_size, seq_len, mask_id)
    model = ProbedInpainter(vocab_size, d_model, num_heads).to(device)

    # 收集探针数据
    probe_history = []

    for epoch in range(epochs):
        model.train()

        input_ids, target_ids = gen.generate_batch(batch_size, max_span)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        z_pred = model.forward(input_ids)

        safe_target_ids = target_ids.clone()
        safe_target_ids[target_ids == -100] = 0
        z_target = model.embedding.weight.data[safe_target_ids].detach()

        valid_mask = (target_ids != -100)
        loss_val, force = compute_elegant_loss_and_force(z_pred, z_target, valid_mask)

        # ---- 使用探针版反传 ----
        if epoch % probe_interval == 0:
            probe_data = model.probed_backward(force, lr, wd)
            probe_data["epoch"] = epoch
            probe_data["loss"] = loss_val

            # 评估准确率
            model.eval()
            with torch.no_grad():
                z_eval = model.forward(input_ids)
                all_embeds = model.embedding.weight.data[:vocab_size]
                valid = (target_ids != -100)
                if valid.any():
                    z_m = z_eval[valid]
                    true_ids = target_ids[valid]
                    if vocab_size <= 1024:
                        d = (z_m.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(-1)
                        pred = d.argmin(dim=-1)
                        acc = float((pred == true_ids).float().mean())
                    else:
                        acc = -1.0
                else:
                    acc = 0.0
            probe_data["accuracy"] = acc
            model.train()

            probe_history.append(probe_data)

            # 打印摘要
            att = probe_data["attenuation"]
            print(
                f"  Epoch {epoch:04d} | Loss: {loss_val:.5f} | Acc: {acc:.1%} | "
                f"Force: {probe_data['force_input']['l2_norm']:.4e} → "
                f"Block: {att['block_ratio']:.4f} → "
                f"Rotary: {att['rotary_ratio']:.4f} → "
                f"Total: {att['total_ratio']:.4f} | "
                f"Δemb/w: {att['delta_vs_weight']:.2e}"
            )
        else:
            # 正常反传 (不记录探针, 节省时间)
            model.probed_backward(force, lr, wd)

    return {
        "config": cfg,
        "probe_history": probe_history,
    }


# =====================================================================
#  可视化报告
# =====================================================================
def plot_probe_report(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
):
    """生成多配置梯度诊断对比图。"""

    n_configs = len(all_results)
    if n_configs == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_configs, 2)))

    for idx, (key, result) in enumerate(sorted(all_results.items())):
        cfg = result["config"]
        history = result["probe_history"]
        if not history:
            continue

        epochs = [p["epoch"] for p in history]
        label = f"{cfg['name']} (V/D={cfg['vocab_size']//cfg['d_model']})"
        color = colors[idx]

        # ---- Panel 1: 力向量 L2 范数 (各层) ----
        ax = axes[0, 0]
        force_norms = [p["force_input"]["l2_norm"] for p in history]
        after_block_norms = [p["after_block"]["l2_norm"] for p in history]
        after_rotary_norms = [p["after_rotary"]["l2_norm"] for p in history]

        ax.semilogy(epochs, force_norms, '-', color=color, alpha=0.8,
                     label=f"{label} input")
        ax.semilogy(epochs, after_block_norms, '--', color=color, alpha=0.5)
        ax.semilogy(epochs, after_rotary_norms, ':', color=color, alpha=0.5)

        # ---- Panel 2: 衰减比 (total = after_rotary / force_input) ----
        ax = axes[0, 1]
        total_ratios = [p["attenuation"]["total_ratio"] for p in history]
        ax.plot(epochs, total_ratios, '-', color=color, label=label)

        # ---- Panel 3: Block 衰减比 ----
        ax = axes[1, 0]
        block_ratios = [p["attenuation"]["block_ratio"] for p in history]
        ax.plot(epochs, block_ratios, '-', color=color, label=label)

        # ---- Panel 4: Rotary 衰减比 ----
        ax = axes[1, 1]
        rotary_ratios = [p["attenuation"]["rotary_ratio"] for p in history]
        ax.plot(epochs, rotary_ratios, '-', color=color, label=label)

        # ---- Panel 5: Embedding 更新量 / 权重 ----
        ax = axes[2, 0]
        delta_ratios = [p["attenuation"]["delta_vs_weight"] for p in history]
        ax.semilogy(epochs, delta_ratios, '-', color=color, label=label)

        # ---- Panel 6: 方向保持度 (cosine similarity) ----
        ax = axes[2, 1]
        dir_block = [p["direction_block"]["cosine_similarity"] for p in history]
        dir_rotary = [p["direction_rotary"]["cosine_similarity"] for p in history]
        ax.plot(epochs, dir_block, '-', color=color, alpha=0.8, label=f"{label} block")
        ax.plot(epochs, dir_rotary, '--', color=color, alpha=0.5)

    # 标注
    axes[0, 0].set_title("Force Vector L2 Norm (solid=input, dash=block, dot=rotary)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("L2 Norm (log)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.2)

    axes[0, 1].set_title("Total Attenuation Ratio (after_rotary / force_input)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Ratio")
    axes[0, 1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.2)

    axes[1, 0].set_title("Block Attenuation (after_block / force_input)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 0].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].set_title("Rotary Attenuation (after_rotary / after_block)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.2)

    axes[2, 0].set_title("Embedding Update / Weight Ratio (log)")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("|delta_emb| / |emb_weight|")
    axes[2, 0].legend(fontsize=7)
    axes[2, 0].grid(True, alpha=0.2)

    axes[2, 1].set_title("Direction Cosine Similarity (solid=block, dash=rotary)")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Cosine Similarity")
    axes[2, 1].set_ylim(-1.05, 1.05)
    axes[2, 1].axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].grid(True, alpha=0.2)

    fig.suptitle("Gradient Probe: Force Vector Attenuation Through Manual Backward",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(output_dir, "GRADIENT_PROBE_REPORT.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Probe] Report saved: {plot_path}")

    return plot_path


# =====================================================================
#  终端摘要报告
# =====================================================================
def print_summary_report(all_results: Dict[str, Dict[str, Any]]):
    """打印梯度诊断的关键发现。"""

    print()
    print("=" * 80)
    print("  Gradient Probe — Summary")
    print("=" * 80)

    header = (f"{'Config':<14} {'V/D':>5} "
              f"{'ForceNorm':>10} {'BlockAtt':>9} {'RotaryAtt':>10} "
              f"{'TotalAtt':>9} {'Δemb/w':>10} "
              f"{'DirBlock':>9} {'DirRotary':>10}")
    print(header)
    print("-" * 80)

    for key in sorted(all_results.keys()):
        result = all_results[key]
        cfg = result["config"]
        history = result["probe_history"]

        if not history:
            continue

        # 取最后 5 个探测点的均值 (训练中后期的稳态行为)
        last_n = history[-5:] if len(history) >= 5 else history

        force_norm = np.mean([p["force_input"]["l2_norm"] for p in last_n])
        block_att = np.mean([p["attenuation"]["block_ratio"] for p in last_n])
        rotary_att = np.mean([p["attenuation"]["rotary_ratio"] for p in last_n])
        total_att = np.mean([p["attenuation"]["total_ratio"] for p in last_n])
        delta_w = np.mean([p["attenuation"]["delta_vs_weight"] for p in last_n])
        dir_block = np.mean([p["direction_block"]["cosine_similarity"] for p in last_n])
        dir_rotary = np.mean([p["direction_rotary"]["cosine_similarity"] for p in last_n])

        ratio = cfg["vocab_size"] / cfg["d_model"]

        print(f"{cfg['name']:<14} {ratio:>4.0f}x "
              f"{force_norm:>10.2e} {block_att:>9.4f} {rotary_att:>10.4f} "
              f"{total_att:>9.4f} {delta_w:>10.2e} "
              f"{dir_block:>9.4f} {dir_rotary:>10.4f}")

    print("=" * 80)

    # ---- 自动诊断 ----
    print()
    print("Diagnosis:")

    for key in sorted(all_results.keys()):
        result = all_results[key]
        cfg = result["config"]
        history = result["probe_history"]
        if not history:
            continue

        last_n = history[-5:] if len(history) >= 5 else history
        total_att = np.mean([p["attenuation"]["total_ratio"] for p in last_n])
        delta_w = np.mean([p["attenuation"]["delta_vs_weight"] for p in last_n])
        dir_block = np.mean([p["direction_block"]["cosine_similarity"] for p in last_n])

        name = cfg["name"]

        if total_att < 0.01:
            print(f"  [{name}] SEVERE: Total attenuation < 1% — "
                  f"gradient vanishing, embedding receives almost no signal.")
        elif total_att < 0.1:
            print(f"  [{name}] WARNING: Total attenuation < 10% — "
                  f"significant gradient decay through backward chain.")
        elif total_att > 10.0:
            print(f"  [{name}] WARNING: Total amplification > 10x — "
                  f"possible gradient explosion.")
        else:
            print(f"  [{name}] OK: Total attenuation {total_att:.2%} — "
                  f"gradient magnitude preserved reasonably.")

        if abs(dir_block) < 0.1:
            print(f"  [{name}] CONCERN: Block direction cosine ~0 — "
                  f"force direction is scrambled by transformer block.")
        elif dir_block < 0:
            print(f"  [{name}] ANOMALY: Block direction cosine negative — "
                  f"block is REVERSING the force direction.")

        if delta_w < 1e-8:
            print(f"  [{name}] SEVERE: Embedding update negligible (Δ/w = {delta_w:.1e}) — "
                  f"embedding is effectively frozen.")

    print()


# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Gradient Probe: diagnose force vector attenuation "
                    "through manual_backward chain"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A", "C", "E"],
        choices=list(PROBE_CONFIGS.keys()),
        help="Configs to probe (default: A C E — spanning the ratio range)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000,
        help="Training epochs per config (default: 1000)"
    )
    parser.add_argument(
        "--probe-interval", type=int, default=50,
        help="Probe every N epochs (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_ANLA_ROOT, "Logs", "gradient_probe"),
        help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto/cpu/cuda"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for config_key in args.configs:
        cfg = PROBE_CONFIGS[config_key].copy()

        try:
            result = probe_single_config(
                cfg=cfg,
                epochs=args.epochs,
                probe_interval=args.probe_interval,
                device=device,
            )
            all_results[config_key] = result
        except Exception as e:
            print(f"\n  [ERROR] Config {config_key} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存原始数据
    # (probe_history 中的数据都是 Python 原生类型, 可直接序列化)
    json_path = os.path.join(args.output_dir, "probe_data.json")
    serializable = {}
    for key, result in all_results.items():
        serializable[key] = {
            "config": result["config"],
            "probe_history": result["probe_history"],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # 报告
    print_summary_report(all_results)
    plot_probe_report(all_results, args.output_dir)

    print(f"Raw data: {json_path}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
