"""
保存位置: Anla/experiments/baselines/real_baseline_v3.py

修正版实数对照实验 v3 — 修复 embedding 坍缩
=========================================================================

v1 的 bug: sqrt(d_model) 缩放导致输出/目标尺度不匹配 → 准确率 ~7%
v2 的 bug: Adam weight_decay 无约束地缩小 embedding 权重
           embedding 范数从 1.0 → 0.003 (坍缩 99.7%)
           所有 token 彼此距离 < 0.01 → 97% NN 一致率是噪声假象

v3 修正:
  1. 分离参数组: embedding 不受 weight_decay 影响
  2. Embedding 稳态机制: 每步更新后归一化到单位范数
     (与 Anla ComplexEmbedding 的 homeostasis 对齐:
      Anla 在每次 _apply_update 后计算 batch_rms 并缩放到 target_energy=1.0)
  3. 移除 sqrt(d_model) 缩放 (v2 已修正)
  4. 其余不变: nn.TransformerEncoderLayer, autograd, Adam

这是公平对照实验的正确版本。

用法:
  python -m Anla.real_baseline_v3 --configs A D
  python -m Anla.real_baseline_v3
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

# [Path Fix] 文件位置: Anla/experiments/baselines/real_baseline_v3.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, "..", ".."))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
#  实验配置
# =====================================================================
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
    variants = []
    for mode in ["same_dim", "same_param"]:
        d_complex = base["d_complex"]
        d_model = d_complex if mode == "same_dim" else 2 * d_complex

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
#  正弦位置编码
# =====================================================================
class SinusoidalPositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

# =====================================================================
#  修正版实数模型 (v3)
# =====================================================================
class RealBaselineInpainter(nn.Module):
    """
    v3 修正版:
    - 不做 sqrt(d_model) 缩放
    - embedding 初始化为单位范数
    - 提供 apply_embedding_homeostasis() 方法, 训练循环中调用

    架构:
      nn.Embedding (单位范数初始化)
      → SinusoidalPositionalEncoding
      → nn.TransformerEncoderLayer (单层, 双向, Pre-LN)
      → 直接输出 (无投影头)
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 dim_feedforward: int = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=256)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        # 初始化 embedding 为单位范数 (与 Anla 对齐)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            norms = self.embedding.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.embedding.weight.div_(norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)           # (B, S, d_model), norm ≈ 1
        z = self.pos_encoder(z)          # 加法
        z = self.transformer_layer(z)    # Pre-LN Transformer
        return z

    @torch.no_grad()
    def apply_embedding_homeostasis(self):
        """
        Embedding 稳态约束 — 与 Anla ComplexEmbedding 的 homeostasis 对齐。

        Anla 的做法 (embedding.py line 102-120):
          1. 计算当前活跃向量的 RMS: sqrt(mean(|z|^2))
          2. 缩放因子 = 1.0 / RMS
          3. 所有活跃向量乘以缩放因子

        我们这里简化为对全部 embedding 做逐向量单位范数归一化。
        这比 Anla 的 batch-RMS 稍严格 (Anla 允许向量间有范数差异),
        但保证不会坍缩, 是公平对照的合理选择。
        """
        norms = self.embedding.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.embedding.weight.data.div_(norms)

# =====================================================================
#  数据生成器 (带 holdout)
# =====================================================================
class RingSpanDataGeneratorWithHoldout:
    def __init__(self, vocab_size, seq_len, mask_id, holdout_frac=0.2, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id

        rng = random.Random(seed)
        all_starts = list(range(vocab_size))
        rng.shuffle(all_starts)
        n_test = max(1, int(vocab_size * holdout_frac))
        self.test_starts_list = all_starts[:n_test]
        self.train_starts = all_starts[n_test:]

    def _generate(self, batch_size, max_span, starts):
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

    def generate_train_batch(self, batch_size, max_span):
        return self._generate(batch_size, max_span, self.train_starts)

    def generate_test_batch(self, batch_size, max_span):
        return self._generate(batch_size, max_span, self.test_starts_list)

# =====================================================================
#  评估: 最近邻匹配
# =====================================================================
@torch.no_grad()
def evaluate_nearest_neighbor(z_pred, target_ids, all_embeds, chunk_size=256):
    valid_mask = (target_ids != -100)
    if not valid_mask.any():
        return 0.0, 0, 0

    z_masked = z_pred[valid_mask]
    true_ids = target_ids[valid_mask]
    n_masked = z_masked.shape[0]

    if n_masked <= chunk_size:
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
def train_single_config(cfg, output_dir, device, log_interval=200,
                        patience=2000, eval_test_interval=500):
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
    print(f"  [REAL v3] {cfg['name']}")
    print(f"  vocab={vocab_size}, d_model={d_model} (real), "
          f"ratio={ratio:.0f}:1, heads={num_heads}")
    print(f"  mode={cfg['mode']}, d_complex_ref={cfg['d_complex_ref']}")
    print(f"  FIX: separate param groups + embedding homeostasis")
    print("=" * 72)

    gen = RingSpanDataGeneratorWithHoldout(
        vocab_size, seq_len, mask_id, cfg["holdout_frac"], seed=42)
    print(f"  Train starts: {len(gen.train_starts)}, "
          f"Test starts: {len(gen.test_starts_list)}")

    model = RealBaselineInpainter(vocab_size, d_model, num_heads).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ★ FIX 1: 分离参数组 — embedding 不受 weight_decay
    # 这与 Anla 一致: Anla 的 weight_decay 在 embedding 中是手动控制的,
    # 且与 homeostasis 配合使用, 不会导致坍缩。
    # 对于公平对照, 实数 embedding 也不应被 Adam 的 L2 regularization 缩小。
    emb_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("embedding.")]

    optimizer = torch.optim.Adam([
        {"params": emb_params,   "lr": lr, "weight_decay": 0.0},   # 无 weight_decay
        {"params": other_params, "lr": lr, "weight_decay": wd},    # 正常 weight_decay
    ])

    mse_fn = nn.MSELoss(reduction='none')

    history = {"epochs": [], "loss": [], "train_acc": [], "test_acc": [],
               "emb_norm_mean": []}
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

        z_pred = model(input_ids)

        # 目标: embedding 原型 (detach, 不通过目标传梯度)
        safe_tgt = target_ids.clone()
        safe_tgt[target_ids == -100] = 0
        z_target = model.embedding.weight[safe_tgt].detach()

        valid_mask = (target_ids != -100)
        loss_elem = mse_fn(z_pred, z_target)
        mask_3d = valid_mask.unsqueeze(-1).float()
        loss_elem = loss_elem * mask_3d
        num_valid = valid_mask.sum().float().clamp(min=1.0)
        loss = loss_elem.sum() / (num_valid * d_model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ★ FIX 2: Embedding 稳态约束
        # 与 Anla ComplexEmbedding 的 homeostasis 对齐
        model.apply_embedding_homeostasis()

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                z_eval = model(input_ids)
                all_embeds = model.embedding.weight[:vocab_size]
                train_acc, n_ok, n_tot = evaluate_nearest_neighbor(
                    z_eval, target_ids, all_embeds)

                # 诊断: embedding 范数
                emb_norms = all_embeds.norm(dim=1)
                emb_norm_mean = emb_norms.mean().item()

            test_acc = -1.0
            if epoch % eval_test_interval == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    ti, tt = gen.generate_test_batch(batch_size, max_span)
                    ti, tt = ti.to(device), tt.to(device)
                    zt = model(ti)
                    test_acc, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds)

            history["epochs"].append(epoch)
            history["loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["emb_norm_mean"].append(emb_norm_mean)

            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""
            print(f"  [{cfg['name']}] Epoch {epoch:05d} | "
                  f"Loss: {loss.item():.6f} | Acc: {train_acc:.2%} ({n_ok}/{n_tot})"
                  f"{test_str} | |emb|: {emb_norm_mean:.4f}")

            if train_acc > best_train_acc or (train_acc == best_train_acc and loss.item() < best_loss):
                best_train_acc = train_acc
                best_loss = loss.item()
                last_best_epoch = epoch
                epochs_no_improve = 0
                # 保存最佳 checkpoint
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "train_acc": train_acc,
                }, os.path.join(output_dir, "best_checkpoint.pth"))
            else:
                epochs_no_improve += log_interval

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if epochs_no_improve >= patience and epoch > 1000:
                print(f"  Early stopping at epoch {epoch} (no improve for {patience} epochs)")
                break

    t_elapsed = time.time() - t_start

    # ---- 最终评估 ----
    model.eval()
    all_embeds = model.embedding.weight[:vocab_size].detach()

    # 多轮测试平均
    test_accs = []
    for _ in range(10):
        ti, tt = gen.generate_test_batch(batch_size, max_span)
        ti, tt = ti.to(device), tt.to(device)
        with torch.no_grad():
            zt = model(ti)
        ta, _, _ = evaluate_nearest_neighbor(zt, tt, all_embeds)
        test_accs.append(ta)
    final_test_acc = float(np.mean(test_accs))
    final_test_std = float(np.std(test_accs))

    nn_rate = ring_neighbor_consistency(all_embeds, vocab_size)

    # Embedding 导出 + TDA
    feat_path = os.path.join(output_dir, "embedding.npz")
    w_np = model.embedding.weight[:vocab_size].detach().cpu().numpy()
    export_embedding_real(model, vocab_size, feat_path)
    tda = quick_tda_audit_real(w_np, k=6)

    # Embedding 统计
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
    print(f"    Emb norm:  {np.mean(emb_norms):.4f} ± {np.std(emb_norms):.4f}")
    if tda.get("available"):
        print(f"    H1:    {tda['h1_count']} features, "
              f"dom={tda['dominant_persistence']:.4f}, "
              f"ratio={tda['dominance_ratio']:.4f}")
    print(f"    Time:  {t_elapsed:.0f}s")

    return result

# =====================================================================
#  对比报告
# =====================================================================
def generate_report(results, output_dir):
    print()
    print("=" * 110)
    print("  Real Baseline v3 (fixed: homeostasis + no emb weight_decay) — Report")
    print("=" * 110)

    header = (f"{'Name':<35} {'V':>5} {'D':>4} {'V/D':>5} {'Mode':<10} "
              f"{'TrainAcc':>9} {'TestAcc':>9} {'NN%':>6} {'|emb|':>7} "
              f"{'H1cnt':>6} {'H1dom':>8} {'H1rat':>7} {'Params':>8} {'Time':>6}")
    print(header)
    print("-" * 110)

    rows = []
    for key in sorted(results.keys()):
        res = results[key]
        cfg = res["config"]
        tda = res["tda_quick"]
        h1c = tda.get("h1_count", -1) if tda.get("available") else -1
        h1d = tda.get("dominant_persistence", -1) if tda.get("available") else -1
        h1r = tda.get("dominance_ratio", -1) if tda.get("available") else -1

        row = {
            "name": cfg["name"], "vocab_size": cfg["vocab_size"],
            "d_model": cfg["d_model"],
            "ratio": cfg["vocab_size"] / cfg["d_model"],
            "mode": cfg["mode"],
            "train_acc": res["best_train_acc"],
            "test_acc": res["final_test_acc"],
            "nn_rate": res["ring_nn_consistency"],
            "emb_norm": res["embedding_norm_mean"],
            "h1_count": h1c, "h1_dominant": h1d, "h1_ratio": h1r,
            "params": res["total_params"], "time": res["training_time_sec"],
        }
        rows.append(row)

        def f(v): return f"{v:.4f}" if v >= 0 else "N/A"
        print(f"{cfg['name']:<35} {cfg['vocab_size']:>5} {cfg['d_model']:>4} "
              f"{row['ratio']:>5.0f} {cfg['mode']:<10} "
              f"{res['best_train_acc']:>8.1%} {res['final_test_acc']:>8.1%} "
              f"{res['ring_nn_consistency']:>5.1%} "
              f"{res['embedding_norm_mean']:>7.4f} "
              f"{str(h1c) if h1c >= 0 else 'N/A':>6} "
              f"{f(h1d):>8} {f(h1r):>7} "
              f"{res['total_params']:>8,} {res['training_time_sec']:>5.0f}s")

    print("=" * 110)

    # ---- 与 Anla 对比 ----
    anla_path = os.path.join(_ANLA_ROOT, "Logs", "capacity_pressure_test", "COMPARISON_REPORT.json")
    if os.path.isfile(anla_path):
        with open(anla_path, "r") as fp:
            anla_data = json.load(fp)

        print()
        print("Direct Comparison: Anla (Complex, manual_backward) vs Real v3 (autograd + homeostasis)")
        print("-" * 100)
        print(f"{'Config':<8} {'Anla Train':>11} {'R(SD) Train':>12} {'R(SP) Train':>12} "
              f"{'Anla NN%':>9} {'R(SD) NN%':>10} {'R(SP) NN%':>10} "
              f"{'R(SP)|emb|':>11}")
        print("-" * 100)

        for bk in sorted(BASE_CONFIGS.keys()):
            ad = anla_data.get(bk, {})
            at = ad.get("best_train_acc", -1)
            an = ad.get("ring_nn_consistency", -1)

            sd = [r for r in rows if r["name"].startswith(f"{bk}_same_dim")]
            sp = [r for r in rows if r["name"].startswith(f"{bk}_same_param")]

            def fmt(v): return f"{v:.1%}" if v >= 0 else "N/A"

            sdt = fmt(sd[0]["train_acc"]) if sd else "N/A"
            sdn = fmt(sd[0]["nn_rate"]) if sd else "N/A"
            spt = fmt(sp[0]["train_acc"]) if sp else "N/A"
            spn = fmt(sp[0]["nn_rate"]) if sp else "N/A"
            spe = f"{sp[0]['emb_norm']:.4f}" if sp else "N/A"

            print(f"{bk:<8} {fmt(at):>11} {sdt:>12} {spt:>12} "
                  f"{fmt(an):>9} {sdn:>10} {spn:>10} {spe:>11}")

        print()

        # 解读
        for bk in sorted(BASE_CONFIGS.keys()):
            ad = anla_data.get(bk, {})
            at = ad.get("best_train_acc", -1)
            an = ad.get("ring_nn_consistency", -1)

            sp = [r for r in rows if r["name"].startswith(f"{bk}_same_param")]
            if not sp or at < 0:
                continue

            spt = sp[0]["train_acc"]
            spn = sp[0]["nn_rate"]
            spe = sp[0]["emb_norm"]

            # 准确率对比
            if at - spt > 0.1:
                acc_verdict = f"Anla >> Real ({at:.1%} vs {spt:.1%})"
            elif spt - at > 0.1:
                acc_verdict = f"Real >> Anla ({spt:.1%} vs {at:.1%})"
            else:
                acc_verdict = f"Comparable ({at:.1%} vs {spt:.1%})"

            # 结构对比
            if spn - an > 0.05:
                nn_verdict = f"Real has MORE structure ({spn:.1%} vs {an:.1%})"
            elif an - spn > 0.05:
                nn_verdict = f"Anla has more structure ({an:.1%} vs {spn:.1%})"
            else:
                nn_verdict = f"Similar structure ({spn:.1%} vs {an:.1%})"

            print(f"  [{bk}] Accuracy: {acc_verdict}")
            print(f"       Structure: {nn_verdict}")
            print(f"       Emb norm:  {spe:.4f} (should be ≈1.0, was 0.003 in v2)")

            if at > spt + 0.1 and spn > an + 0.05:
                print(f"       ★ CRITICAL: Anla learns the TASK better, "
                      f"but Real learns the STRUCTURE better.")
                print(f"         This confirms manual_backward gradient quality issue.")
            elif abs(at - spt) < 0.1 and abs(spn - an) < 0.05:
                print(f"       → Both comparable: single-block architecture is the bottleneck.")
            elif spt > at + 0.1 and spn > an + 0.05:
                print(f"       → Real dominates: autograd + real valued is strictly better here.")
            print()

    # 保存
    report_path = os.path.join(output_dir, "REAL_BASELINE_V3_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {report_path}")

# =====================================================================
#  v2 对比 (说明修复效果)
# =====================================================================
def compare_with_v2(results, output_dir):
    v2_path = os.path.join(_ANLA_ROOT, "Logs", "real_baseline_v2", "REAL_BASELINE_V2_REPORT.json")
    if not os.path.isfile(v2_path):
        return

    with open(v2_path) as f:
        v2_rows = json.load(f)

    v3_rows = []
    for key in sorted(results.keys()):
        res = results[key]
        cfg = res["config"]
        v3_rows.append({
            "name": cfg["name"],
            "train_acc": res["best_train_acc"],
            "nn_rate": res["ring_nn_consistency"],
            "emb_norm": res["embedding_norm_mean"],
        })

    print()
    print("=" * 80)
    print("  v2 → v3 Fix Verification")
    print("=" * 80)
    print(f"{'Name':<35} {'v2 Acc':>8} {'v3 Acc':>8} {'v2 NN%':>7} {'v3 NN%':>7} "
          f"{'v2 |emb|':>9} {'v3 |emb|':>9}")
    print("-" * 80)

    for v3r in v3_rows:
        v2match = [r for r in v2_rows if r["name"] == v3r["name"]]
        if v2match:
            v2r = v2match[0]
            print(f"{v3r['name']:<35} "
                  f"{v2r['train_acc']:>7.1%} {v3r['train_acc']:>7.1%} "
                  f"{v2r['nn_rate']:>6.1%} {v3r['nn_rate']:>6.1%} "
                  f"{v2r.get('emb_norm', 0):>9.4f} {v3r['emb_norm']:>9.4f}")
        else:
            print(f"{v3r['name']:<35} {'N/A':>8} {v3r['train_acc']:>7.1%} "
                  f"{'N/A':>7} {v3r['nn_rate']:>6.1%} "
                  f"{'N/A':>9} {v3r['emb_norm']:>9.4f}")

    print()
    print("  v2 had embedding collapse (|emb| → 0.003), making all metrics unreliable.")
    print("  v3 maintains |emb| ≈ 1.0 via homeostasis, giving valid comparisons.")
    print()

# =====================================================================
#  主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Real baseline v3 (fixed: homeostasis + separate param groups)")
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E"],
                        choices=list(BASE_CONFIGS.keys()))
    parser.add_argument("--modes", nargs="+", default=["same_dim", "same_param"],
                        choices=["same_dim", "same_param"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_ANLA_ROOT, "Logs", "real_baseline_v3"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for ck in args.configs:
        base = BASE_CONFIGS[ck]
        for cfg in expand_config(ck, base):
            if cfg["mode"] not in args.modes:
                continue
            if args.epochs is not None:
                cfg["epochs"] = args.epochs

            config_dir = os.path.join(args.output_dir, f"config_{cfg['name']}")
            try:
                result = train_single_config(cfg, config_dir, device,
                                             args.log_interval, args.patience)
                results[cfg["name"]] = result
            except Exception as e:
                print(f"\n  [ERROR] {cfg['name']}: {e}")
                import traceback; traceback.print_exc()

    if results:
        generate_report(results, args.output_dir)
        compare_with_v2(results, args.output_dir)

if __name__ == "__main__":
    main()

