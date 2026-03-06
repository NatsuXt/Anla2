"""
保存位置: Anla/experiments/baselines/real_baseline_comparison.py

实数对照实验 — 验证 Anla 观察到的现象是复数特有的还是架构通用的
=========================================================================

核心问题:
    Anla 在容量压力测试中没有在 embedding 中涌现环拓扑结构。
    这是因为:
      (A) 复数空间的几何性质不同?
      (B) manual_backward 的精度不如 autograd?
      (C) 单 block Transformer 在任何值域下都学不出结构?
      (D) 环形任务的上下文多样性不足?

    只有实数对照组能区分这些解释。

实验设计:
    使用 PyTorch 原生组件 (nn.Embedding, nn.TransformerEncoderLayer, autograd),
    跑完全相同的环形 span masking 任务, 完全相同的 vocab/d_model 配置矩阵。

    额外设置两种维度对齐方式:
      - "same_dim":  实数 d_model = 复数 d_model (同维度, 实数参数量是复数的一半)
      - "same_param": 实数 d_model = 2 × 复数 d_model (同参数量)

    这两种对齐方式能进一步区分:
      如果 same_dim 就能学出结构 → 复数的额外自由度不是必需的
      如果只有 same_param 能学出 → 复数在"每维度两个自由度"上没有额外优势
      如果两者都学不出 → 问题在任务或架构深度, 不在值域

用法:
    python -m Anla.real_baseline_comparison
    python -m Anla.real_baseline_comparison --configs A D
    python -m Anla.real_baseline_comparison --configs A B C D E --epochs 10000
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------------
# [Path Fix] 文件位置: Anla/experiments/baselines/real_baseline_comparison.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))


# =====================================================================
#  实验配置 (与 capacity_pressure_test.py 对齐)
# =====================================================================
#  每个配置产生两个实数变体:
#    - same_dim:  d_real = d_complex
#    - same_param: d_real = 2 * d_complex (参数量对齐)

BASE_CONFIGS = {
    "A": {"vocab_size": 64,   "d_complex": 64, "num_heads": 4,
           "batch_size": 16,  "epochs": 5000},
    "B": {"vocab_size": 256,  "d_complex": 64, "num_heads": 4,
           "batch_size": 32,  "epochs": 10000},
    "C": {"vocab_size": 512,  "d_complex": 32, "num_heads": 4,
           "batch_size": 32,  "epochs": 15000},
    "D": {"vocab_size": 1024, "d_complex": 32, "num_heads": 4,
           "batch_size": 64,  "epochs": 20000},
    "E": {"vocab_size": 1024, "d_complex": 16, "num_heads": 2,
           "batch_size": 64,  "epochs": 20000},
}


def expand_config(key: str, base: dict) -> List[dict]:
    """
    从一个基础配置生成两个实数变体。

    same_dim:  d_model = d_complex       (同维度, 实数参数更少)
    same_param: d_model = 2 * d_complex  (同参数量)
    """
    variants = []
    for mode in ["same_dim", "same_param"]:
        d_complex = base["d_complex"]

        if mode == "same_dim":
            d_model = d_complex
        else:
            d_model = 2 * d_complex

        # num_heads 需要能整除 d_model
        num_heads = base["num_heads"]
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        variants.append({
            "name": f"{key}_{mode}_v{base['vocab_size']}_d{d_model}",
            "base_key": key,
            "mode": mode,
            "vocab_size": base["vocab_size"],
            "d_model": d_model,
            "d_complex_ref": d_complex,
            "num_heads": num_heads,
            "seq_len": 32,
            "max_span_length": 5,
            "batch_size": base["batch_size"],
            "lr": 0.001,
            "weight_decay": 1e-4,
            "epochs": base["epochs"],
            "holdout_frac": 0.2,
        })

    return variants


# =====================================================================
#  正弦位置编码 (标准实数版, 不用 rotary 以避免引入额外变量)
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦位置编码, 与 "Attention Is All You Need" 一致。
    """
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# =====================================================================
#  实数基线模型
# =====================================================================
class RealBaselineInpainter(nn.Module):
    """
    标准 PyTorch 实数 Transformer 模型, 结构对标 AnlaManifoldInpainter:
      - nn.Embedding (实数)
      - SinusoidalPositionalEncoding
      - nn.TransformerEncoderLayer (单层, 双向注意力)
      - 无投影头, 直接输出 (Batch, Seq, d_model)

    使用 PyTorch autograd, 不使用 manual_backward。
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 dim_feedforward: int = None):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model  # 标准 Transformer 惯例

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=256)

        # 单层 Transformer encoder (双向注意力)
        # batch_first=True 使接口与 Anla 一致
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,       # 与 Anla 一致, 无 dropout
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN, 更稳定
        )

        # 初始化 embedding 为单位范数 (与 Anla 的 ComplexEmbedding 初始化对齐)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            norms = self.embedding.weight.norm(dim=1, keepdim=True)
            self.embedding.weight.div_(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Seq) — 整数 token ids
        返回: (Batch, Seq, d_model) — 实数状态向量
        """
        z = self.embedding(x) * math.sqrt(self.d_model)  # 标准缩放
        z = self.pos_encoder(z)
        z = self.transformer_layer(z)
        return z


# =====================================================================
#  数据生成器 (与 capacity_pressure_test.py 完全一致)
# =====================================================================
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

    def _generate_batch(self, batch_size, starts, max_span):
        inputs, targets = [], []
        for _ in range(batch_size):
            start = random.choice(starts)
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

    def generate_train_batch(self, batch_size, max_span=5):
        return self._generate_batch(batch_size, self.train_starts, max_span)

    def generate_test_batch(self, batch_size, max_span=5):
        return self._generate_batch(batch_size, self.test_starts_list, max_span)


# =====================================================================
#  最近邻评估
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

    if vocab_size <= 2048:
        # (N, 1, d) - (1, V, d) → (N, V)
        dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)).pow(2).sum(dim=-1)
        pred_ids = dists.argmin(dim=-1)
    else:
        pred_ids = torch.empty(n_masked, dtype=torch.long, device=z_masked.device)
        for i in range(0, n_masked, chunk_size):
            end = min(i + chunk_size, n_masked)
            chunk = z_masked[i:end]
            d = (chunk.unsqueeze(1) - all_embeds.unsqueeze(0)).pow(2).sum(dim=-1)
            pred_ids[i:end] = d.argmin(dim=-1)

    n_correct = (pred_ids == true_ids).sum().item()
    n_total = true_ids.shape[0]
    return float(n_correct / n_total) if n_total > 0 else 0.0, n_correct, n_total


# =====================================================================
#  环邻居一致率
# =====================================================================
@torch.no_grad()
def ring_neighbor_consistency(all_embeds, vocab_size):
    w = all_embeds[:vocab_size]
    n = w.shape[0]
    nn_ids = torch.empty(n, dtype=torch.long, device=w.device)
    for i in range(n):
        dists = (w[i].unsqueeze(0) - w).pow(2).sum(dim=-1)
        dists[i] = float("inf")
        nn_ids[i] = dists.argmin()
    nn_ids = nn_ids.cpu().numpy()
    token_ids = np.arange(n)
    is_ring = (nn_ids == (token_ids - 1) % n) | (nn_ids == (token_ids + 1) % n)
    return float(is_ring.mean())


# =====================================================================
#  Embedding 导出 + 轻量 TDA
# =====================================================================
def export_embedding_real(model, vocab_size, path):
    w = model.embedding.weight.data[:vocab_size].detach().cpu().numpy()
    w = np.asarray(w, dtype=np.float64)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, x_real=w)
    return w.shape


def quick_tda_audit_real(w_np, k=6):
    try:
        from scipy.sparse import lil_matrix
        from scipy.sparse.csgraph import shortest_path
        from scipy.spatial.distance import pdist, squareform
        from ripser import ripser
    except ImportError:
        return {"available": False, "reason": "ripser/scipy not installed"}

    x = w_np.astype(np.float64)
    n = x.shape[0]
    De = squareform(pdist(x, metric="euclidean"))

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

    result = ripser(Dg, distance_matrix=True, maxdim=1)
    dgm1 = result["dgms"][1]

    if dgm1.size == 0:
        return {"available": True, "h1_count": 0,
                "dominant_persistence": 0.0, "dominance_ratio": 0.0}

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
#  单配置训练
# =====================================================================
def train_single_config(
    cfg: dict,
    output_dir: str,
    device: torch.device,
    log_interval: int = 200,
    patience: int = 2000,
    eval_test_interval: int = 500,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)

    vocab_size = cfg["vocab_size"]
    d_model = cfg["d_model"]
    num_heads = cfg["num_heads"]
    seq_len = cfg["seq_len"]
    max_span = cfg["max_span_length"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    epochs = cfg["epochs"]
    mask_id = vocab_size

    ratio = vocab_size / d_model

    print()
    print("=" * 72)
    print(f"  [REAL] {cfg['name']}")
    print(f"  vocab={vocab_size}, d_model={d_model} (real), "
          f"ratio={ratio:.0f}:1, heads={num_heads}")
    print(f"  mode={cfg['mode']}, d_complex_ref={cfg['d_complex_ref']}")
    print(f"  epochs={epochs}, batch={batch_size}, lr={lr}")
    print("=" * 72)

    # ---- 数据 ----
    gen = RingSpanDataGeneratorWithHoldout(
        vocab_size, seq_len, mask_id, cfg["holdout_frac"], seed=42)
    print(f"  Train starts: {len(gen.train_starts)}, "
          f"Test starts: {len(gen.test_starts_list)}")

    # ---- 模型 ----
    model = RealBaselineInpainter(vocab_size, d_model, num_heads).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- 优化器: Adam, 与 Anla 的 lr 和 wd 对齐 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # ---- Loss: MSE (实数版 L_Elegant 的等价物) ----
    # 只在 masked 位置计算, 目标是 embedding 原型
    mse_loss = nn.MSELoss(reduction='none')

    # ---- 记录 ----
    history = {"epochs": [], "loss": [], "train_acc": [], "test_acc": []}
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_loss = float("inf")
    epochs_no_improve = 0
    last_best_epoch = 0

    t_start = time.time()

    for epoch in range(epochs):
        model.train()

        input_ids, target_ids = gen.generate_train_batch(batch_size, max_span)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 前向
        z_pred = model(input_ids)  # (B, S, d_model)

        # 目标: embedding 原型 (detach, 不要通过目标传梯度)
        safe_tgt = target_ids.clone()
        safe_tgt[target_ids == -100] = 0
        z_target = model.embedding.weight[safe_tgt].detach()

        # masked MSE loss
        valid_mask = (target_ids != -100)  # (B, S)
        loss_elem = mse_loss(z_pred, z_target)  # (B, S, d_model)
        mask_3d = valid_mask.unsqueeze(-1).float()
        loss_elem = loss_elem * mask_3d
        num_valid = valid_mask.sum().float().clamp(min=1.0)
        loss = loss_elem.sum() / (num_valid * d_model)

        # 反向 + 更新 (标准 autograd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 评估 ----
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                z_eval = model(input_ids)
                all_embeds = model.embedding.weight[:vocab_size]
                train_acc, n_ok, n_tot = evaluate_nearest_neighbor(
                    z_eval, target_ids, all_embeds)

                test_acc = -1.0
                if epoch % eval_test_interval == 0 or epoch == epochs - 1:
                    ti, tt = gen.generate_test_batch(batch_size, max_span)
                    ti, tt = ti.to(device), tt.to(device)
                    zt = model(ti)
                    test_acc, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds)

            history["epochs"].append(epoch)
            history["loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""
            print(f"  [{cfg['name']}] Epoch {epoch:05d} | "
                  f"Loss: {loss.item():.6f} | Train: {train_acc:.2%}{test_str}")

            if train_acc > best_train_acc or \
               (train_acc == best_train_acc and loss.item() < best_loss):
                best_train_acc = train_acc
                best_loss = loss.item()
                last_best_epoch = epoch
                epochs_no_improve = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                }, os.path.join(output_dir, "best_checkpoint.pth"))
            else:
                epochs_no_improve += log_interval

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if epochs_no_improve >= patience and best_train_acc > 0.5:
                print(f"  [Early Stop] at epoch {epoch} (best @ {last_best_epoch})")
                break

            model.train()

    t_elapsed = time.time() - t_start

    # ---- 最终评估 ----
    model.eval()
    with torch.no_grad():
        all_embeds = model.embedding.weight[:vocab_size]

        test_accs = []
        for _ in range(10):
            ti, tt = gen.generate_test_batch(batch_size, max_span)
            ti, tt = ti.to(device), tt.to(device)
            zt = model(ti)
            ta, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds)
            test_accs.append(ta)
        final_test_acc = float(np.mean(test_accs))
        final_test_std = float(np.std(test_accs))

        nn_rate = ring_neighbor_consistency(all_embeds, vocab_size)

    # ---- Embedding 导出 + TDA ----
    feat_path = os.path.join(output_dir, "embedding.npz")
    w_np = model.embedding.weight[:vocab_size].detach().cpu().numpy()
    export_embedding_real(model, vocab_size, feat_path)
    tda = quick_tda_audit_real(w_np, k=6)

    # ---- Embedding 统计 ----
    emb_norms = np.linalg.norm(w_np, axis=1)

    result = {
        "config": cfg,
        "total_params": total_params,
        "training_time_sec": round(t_elapsed, 1),
        "final_epoch": epoch,
        "best_train_acc": best_train_acc,
        "best_train_epoch": last_best_epoch,
        "final_test_acc": final_test_acc,
        "final_test_std": final_test_std,
        "best_test_acc": best_test_acc,
        "ring_nn_consistency": nn_rate,
        "embedding_norm_mean": float(np.mean(emb_norms)),
        "embedding_norm_std": float(np.std(emb_norms)),
        "tda_quick": tda,
        "features_path": feat_path,
        "history": history,
    }

    with open(os.path.join(output_dir, "training_log.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  [{cfg['name']}] Done:")
    print(f"    Train Acc: {best_train_acc:.2%} (epoch {last_best_epoch})")
    print(f"    Test Acc:  {final_test_acc:.2%} ± {final_test_std:.2%}")
    print(f"    Ring NN:   {nn_rate:.1%}")
    if tda.get("available"):
        print(f"    H1 count:  {tda['h1_count']}")
        print(f"    H1 dom:    {tda['dominant_persistence']:.4f}")
        print(f"    H1 ratio:  {tda['dominance_ratio']:.4f}")
    print(f"    Time:      {t_elapsed:.0f}s")

    return result


# =====================================================================
#  对比报告
# =====================================================================
def generate_comparison_report(results: dict, output_dir: str):

    print()
    print("=" * 100)
    print("  Real Baseline — Comparison Report")
    print("=" * 100)

    header = (f"{'Name':<35} {'V':>5} {'D':>4} {'V/D':>5} {'Mode':<10} "
              f"{'TrainAcc':>9} {'TestAcc':>9} {'NN%':>6} "
              f"{'H1cnt':>6} {'H1dom':>8} {'Ratio':>7} {'Params':>8} {'Time':>6}")
    print(header)
    print("-" * 100)

    rows = []
    for key in sorted(results.keys()):
        res = results[key]
        cfg = res["config"]
        tda = res["tda_quick"]
        h1_cnt = tda.get("h1_count", -1) if tda.get("available") else -1
        h1_dom = tda.get("dominant_persistence", -1) if tda.get("available") else -1
        h1_rat = tda.get("dominance_ratio", -1) if tda.get("available") else -1

        row = {
            "name": cfg["name"],
            "vocab_size": cfg["vocab_size"],
            "d_model": cfg["d_model"],
            "ratio": cfg["vocab_size"] / cfg["d_model"],
            "mode": cfg["mode"],
            "train_acc": res["best_train_acc"],
            "test_acc": res["final_test_acc"],
            "nn_rate": res["ring_nn_consistency"],
            "h1_count": h1_cnt,
            "h1_dominant": h1_dom,
            "h1_ratio": h1_rat,
            "params": res["total_params"],
            "time": res["training_time_sec"],
        }
        rows.append(row)

        h1c = str(h1_cnt) if h1_cnt >= 0 else "N/A"
        h1d = f"{h1_dom:.3f}" if h1_dom >= 0 else "N/A"
        h1r = f"{h1_rat:.3f}" if h1_rat >= 0 else "N/A"

        print(f"{cfg['name']:<35} {cfg['vocab_size']:>5} {cfg['d_model']:>4} "
              f"{row['ratio']:>5.0f} {cfg['mode']:<10} "
              f"{res['best_train_acc']:>8.1%} {res['final_test_acc']:>8.1%} "
              f"{res['ring_nn_consistency']:>5.1%} "
              f"{h1c:>6} {h1d:>8} {h1r:>7} "
              f"{res['total_params']:>8,} {res['training_time_sec']:>5.0f}s")

    print("=" * 100)

    # ---- 保存 ----
    report_path = os.path.join(output_dir, "REAL_BASELINE_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # ---- 自动对比分析 ----
    print()
    print("Analysis vs Anla (Complex):")
    print("-" * 60)

    # 加载 Anla 结果 (如果存在)
    anla_report_path = os.path.join(_ANLA_ROOT, "Logs", "capacity_pressure_test", "COMPARISON_REPORT.json")
    if os.path.isfile(anla_report_path):
        with open(anla_report_path, "r", encoding="utf-8") as f:
            anla_data = json.load(f)

        print(f"\n{'Config':<8} {'Anla(C) Train':>14} {'Real(SD) Train':>15} "
              f"{'Real(SP) Train':>15} {'Anla NN%':>9} {'Real(SD) NN%':>12} "
              f"{'Real(SP) NN%':>12}")
        print("-" * 90)

        for base_key in sorted(BASE_CONFIGS.keys()):
            anla_row = anla_data.get(base_key, {})
            anla_train = anla_row.get("best_train_acc", -1)
            anla_nn = anla_row.get("ring_nn_consistency", -1)

            sd_rows = [r for r in rows if r["name"].startswith(f"{base_key}_same_dim")]
            sp_rows = [r for r in rows if r["name"].startswith(f"{base_key}_same_param")]

            sd_train = sd_rows[0]["train_acc"] if sd_rows else -1
            sp_train = sp_rows[0]["train_acc"] if sp_rows else -1
            sd_nn = sd_rows[0]["nn_rate"] if sd_rows else -1
            sp_nn = sp_rows[0]["nn_rate"] if sp_rows else -1

            def fmt(v):
                return f"{v:.1%}" if v >= 0 else "N/A"

            print(f"{base_key:<8} {fmt(anla_train):>14} {fmt(sd_train):>15} "
                  f"{fmt(sp_train):>15} {fmt(anla_nn):>9} {fmt(sd_nn):>12} "
                  f"{fmt(sp_nn):>12}")

        print()

        # 判断趋势
        for base_key in sorted(BASE_CONFIGS.keys()):
            anla_row = anla_data.get(base_key, {})
            anla_train = anla_row.get("best_train_acc", -1)
            anla_nn = anla_row.get("ring_nn_consistency", -1)

            sp_rows = [r for r in rows if r["name"].startswith(f"{base_key}_same_param")]
            if not sp_rows:
                continue

            sp_train = sp_rows[0]["train_acc"]
            sp_nn = sp_rows[0]["nn_rate"]

            if anla_train >= 0 and sp_train >= 0:
                if sp_train > anla_train + 0.1:
                    print(f"  [{base_key}] Real >> Anla in accuracy "
                          f"({sp_train:.1%} vs {anla_train:.1%}) — "
                          f"autograd advantage or architectural difference.")
                elif anla_train > sp_train + 0.1:
                    print(f"  [{base_key}] Anla >> Real in accuracy "
                          f"({anla_train:.1%} vs {sp_train:.1%}) — "
                          f"complex representation advantage.")
                else:
                    print(f"  [{base_key}] Comparable accuracy "
                          f"({anla_train:.1%} vs {sp_train:.1%}).")

            if sp_nn > 0.05 and anla_nn >= 0:
                if sp_nn > anla_nn * 2:
                    print(f"    → Real shows HIGHER ring NN consistency "
                          f"({sp_nn:.1%} vs {anla_nn:.1%}) — "
                          f"structure emerging in real but not complex!")
                elif anla_nn > sp_nn * 2:
                    print(f"    → Anla shows higher ring NN consistency.")

    else:
        print("  (Anla capacity test results not found at "
              f"{anla_report_path}, skipping direct comparison)")

    print()
    print(f"Full report: {report_path}")


# =====================================================================
#  可视化
# =====================================================================
def plot_comparison(results: dict, output_dir: str):
    """生成训练曲线对比图。"""

    # 按 base_key 分组
    groups = {}
    for key, res in results.items():
        bk = res["config"]["base_key"]
        if bk not in groups:
            groups[bk] = {}
        groups[bk][res["config"]["mode"]] = res

    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 5 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, 2)

    colors = {"same_dim": "#2196F3", "same_param": "#FF5722"}

    for row_idx, (bk, group) in enumerate(sorted(groups.items())):
        # Left: Loss
        ax = axes[row_idx, 0]
        for mode, res in sorted(group.items()):
            h = res["history"]
            if h["epochs"]:
                ax.semilogy(h["epochs"], h["loss"], '-',
                           color=colors[mode], label=f"{mode} (d={res['config']['d_model']})")
        ax.set_title(f"Config {bk}: Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (log)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Right: Accuracy
        ax = axes[row_idx, 1]
        for mode, res in sorted(group.items()):
            h = res["history"]
            if h["epochs"]:
                ax.plot(h["epochs"], h["train_acc"], '-',
                       color=colors[mode], label=f"{mode} train")
                test_epochs = [e for e, t in zip(h["epochs"], h["test_acc"]) if t >= 0]
                test_vals = [t for t in h["test_acc"] if t >= 0]
                if test_epochs:
                    ax.plot(test_epochs, test_vals, '--',
                           color=colors[mode], alpha=0.6, label=f"{mode} test")
        ax.set_title(f"Config {bk}: Accuracy (V={group[list(group.keys())[0]]['config']['vocab_size']})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Real Baseline: Training Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(output_dir, "REAL_BASELINE_CURVES.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] Saved: {plot_path}")


# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Real-valued baseline comparison for Anla ring masking task"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["A", "B", "C", "D", "E"],
        choices=list(BASE_CONFIGS.keys()),
        help="Base configs to test (default: all)"
    )
    parser.add_argument(
        "--modes", nargs="+", default=["same_dim", "same_param"],
        choices=["same_dim", "same_param"],
        help="Dimension alignment modes (default: both)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epoch count for all configs"
    )
    parser.add_argument(
        "--patience", type=int, default=2000,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--log-interval", type=int, default=200,
        help="Log interval"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_ANLA_ROOT, "Logs", "real_baseline"),
        help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for config_key in args.configs:
        base = BASE_CONFIGS[config_key]
        variants = expand_config(config_key, base)

        for cfg in variants:
            if cfg["mode"] not in args.modes:
                continue
            if args.epochs is not None:
                cfg["epochs"] = args.epochs

            config_dir = os.path.join(args.output_dir, f"config_{cfg['name']}")
            result_key = cfg["name"]

            try:
                result = train_single_config(
                    cfg=cfg,
                    output_dir=config_dir,
                    device=device,
                    log_interval=args.log_interval,
                    patience=args.patience,
                )
                results[result_key] = result
            except Exception as e:
                print(f"\n  [ERROR] {cfg['name']} failed: {e}")
                import traceback
                traceback.print_exc()

    if results:
        generate_comparison_report(results, args.output_dir)
        plot_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
