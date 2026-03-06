"""
保存位置: Anla/visualization/visualize_config_B.py
Config B 全量可视化分析
用法:
    python -m Anla.visualization.visualize_config_B
    python -m Anla.visualization.visualize_config_B --data-dir Logs/config_B_analysis/config_B_v256_d64
"""
import argparse, json, os, sys, warnings
from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "figure.facecolor": "white",
})

# [Path Fix] 文件位置: Anla/visualization/visualize_config_B.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# =============================================================================
#  Data Loading
# =============================================================================
def load_training_log(data_dir):
    path = os.path.join(data_dir, "training_log.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No log: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_embedding(data_dir):
    path = os.path.join(data_dir, "ring_features.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embedding: {path}")
    return np.load(path)["z"]

def load_checkpoint(data_dir, device="cpu"):
    import torch
    path = os.path.join(data_dir, "best_checkpoint.pth")
    if not os.path.exists(path):
        print(f"  [WARN] No checkpoint: {path}"); return None
    return torch.load(path, map_location=device, weights_only=False)

def reconstruct_model(checkpoint, device="cpu"):
    try:
        import torch
        from Anla.experiments.capacity.capacity_pressure_test_v4 import AnlaManifoldInpainter
        cfg = checkpoint["config"]
        model = AnlaManifoldInpainter(cfg["vocab_size"], cfg["d_model"], cfg["num_heads"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); return model
    except Exception as e:
        print(f"  [WARN] Model failed: {e}"); return None

# =============================================================================
#  Fig 1: Training Dynamics (6 panels)
# =============================================================================
def plot_training_dynamics(history, config, result, save_path):
    epochs = history["epochs"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    vd = config["vocab_size"] // config["d_model"]
    fig.suptitle(f"Training Dynamics - {config['name']} (V={config['vocab_size']}, D={config['d_model']}, V/D={vd})", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color="royalblue", alpha=0.8, linewidth=0.8, label="L_BE")
    if "loss_elegant" in history and any(v > 0 for v in history["loss_elegant"]):
        ax.plot(epochs, history["loss_elegant"], color="coral", alpha=0.7, linewidth=0.8, linestyle="--", label="L_Elegant")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("(1) Loss")
    ax.legend(fontsize=8); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], color="forestgreen", linewidth=1.0)
    bta = result.get("best_train_acc", 0)
    ax.axhline(y=bta, color="red", linestyle=":", alpha=0.5, label=f"Best: {bta:.1%}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("(2) Train Accuracy")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    te = [e for e, t in zip(epochs, history["test_acc"]) if t >= 0]
    ta = [t for t in history["test_acc"] if t >= 0]
    ax.plot(te, ta, color="darkorange", linewidth=1.0, marker=".", markersize=3)
    bte = result.get("best_test_acc", 0)
    ax.axhline(y=bte, color="red", linestyle=":", alpha=0.5, label=f"Best: {bte:.1%}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("(3) Test Accuracy")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["rho"], color="mediumpurple", linewidth=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("rho"); ax.set_title("(4) Learning Progress rho(t)")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, "rho=(L0-L_smooth)/L0\n0=no progress, 1=converged", transform=ax.transAxes, fontsize=7, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax = axes[1, 1]
    ax.plot(epochs, history["emb_rms"], color="teal", linewidth=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMS(|w|)"); ax.set_title("(5) Embedding RMS")
    ax.grid(True, alpha=0.3); ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="Unit sphere"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(epochs, history["train_acc"], color="forestgreen", linewidth=0.8, alpha=0.7, label="Train")
    if te: ax.plot(te, ta, color="darkorange", linewidth=0.8, alpha=0.7, label="Test")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("(6) Generalization Gap")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Training dynamics: {save_path}")

# =============================================================================
#  Fig 2: Boltzmann Diagnostics (4 panels)
# =============================================================================
def plot_boltzmann_diagnostics(history, config, save_path):
    epochs = history["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Boltzmann-Elegant Diagnostics - {config['name']}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]; tau_vals = history.get("tau", [])
    if tau_vals and any(v > 0 for v in tau_vals):
        ax.plot(epochs, tau_vals, color="crimson", linewidth=1.0)
        ax.set_ylabel("tau=std(E_k)"); ax.set_title("(1) Temperature tau")
        ax.text(0.02, 0.95, "tau=std(E_k)\nhigh->flat\nlow->sharp", transform=ax.transAxes, fontsize=7, va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
    else:
        ax.text(0.5, 0.5, "tau unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray"); ax.set_title("(1) Temperature tau")
    ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]; p_vals = history.get("p_target_mean", [])
    if p_vals and any(v > 0 for v in p_vals):
        ax.plot(epochs, p_vals, color="royalblue", linewidth=1.0)
        rvl = 1.0 / config["vocab_size"]
        ax.axhline(y=rvl, color="red", linestyle=":", alpha=0.5, label=f"Random: 1/V={rvl:.4f}")
        ax.set_ylabel("p_target"); ax.set_title("(2) Target Probability"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "p_target unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray"); ax.set_title("(2) Target Probability")
    ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]; neg_vals = history.get("negative_margin_ratio", [])
    if neg_vals and any(v >= 0 for v in neg_vals):
        ve = [e for e, n in zip(epochs, neg_vals) if n >= 0]; vn = [n for n in neg_vals if n >= 0]
        ax.plot(ve, [n*100 for n in vn], color="orangered", linewidth=1.0)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, label="50%=chaos")
        ax.set_ylabel("neg% (%)"); ax.set_title("(3) Negative Margin Ratio"); ax.set_ylim(-5, 105); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "neg% unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray"); ax.set_title("(3) Negative Margin Ratio")
    ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]; gap_vals = history.get("energy_gap", [])
    if gap_vals and any(v > 0 for v in gap_vals):
        ax.plot(epochs, gap_vals, color="seagreen", linewidth=1.0)
        ax.set_ylabel("|E_nw-E_tgt|"); ax.set_title("(4) Energy Gap")
    else:
        ax.text(0.5, 0.5, "Energy Gap unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray"); ax.set_title("(4) Energy Gap")
    ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Boltzmann diagnostics: {save_path}")

# =============================================================================
#  Fig 3: PCA (3 panels)
# =============================================================================
def plot_embedding_pca(z, config, save_path):
    V, D = z.shape; z_real = np.column_stack([z.real, z.imag])
    z_c = z_real - z_real.mean(axis=0, keepdims=True)
    cov = z_c.T @ z_c / V; eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]; eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    pc = z_c @ eigvecs[:, :3]; expl = eigvals[:3] / eigvals.sum() * 100
    colors = np.arange(V); cmap = plt.cm.hsv

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Embedding PCA - {config['name']} (V={V}, D={D})", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 1)
    sc = ax1.scatter(pc[:,0], pc[:,1], c=colors, cmap=cmap, s=8, alpha=0.8, edgecolors="none")
    ax1.set_xlabel(f"PC1 ({expl[0]:.1f}%)"); ax1.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    ax1.set_title("(a) PCA 2D - Token ID"); plt.colorbar(sc, ax=ax1, label="Token ID", shrink=0.8)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(1, 3, 2)
    for k in range(V):
        kn = (k+1)%V; ax2.plot([pc[k,0],pc[kn,0]], [pc[k,1],pc[kn,1]], color="gray", linewidth=0.3, alpha=0.5)
    ax2.scatter(pc[:,0], pc[:,1], c=colors, cmap=cmap, s=8, alpha=0.9, edgecolors="none", zorder=5)
    ax2.set_xlabel(f"PC1 ({expl[0]:.1f}%)"); ax2.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    ax2.set_title("(b) PCA 2D - Ring"); ax2.set_aspect("equal"); ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(pc[:,0], pc[:,1], pc[:,2], c=colors, cmap=cmap, s=6, alpha=0.7)
    for k in range(V):
        kn = (k+1)%V; ax3.plot([pc[k,0],pc[kn,0]], [pc[k,1],pc[kn,1]], [pc[k,2],pc[kn,2]], color="gray", linewidth=0.2, alpha=0.4)
    ax3.set_xlabel(f"PC1 ({expl[0]:.1f}%)"); ax3.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    ax3.set_zlabel(f"PC3 ({expl[2]:.1f}%)"); ax3.set_title("(c) PCA 3D"); ax3.view_init(elev=25, azim=45)
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Embedding PCA: {save_path}")

# =============================================================================
#  Fig 4: Phase Structure (4 panels)
# =============================================================================
def plot_phase_structure(z, config, save_path):
    V, D = z.shape; phases = np.angle(z)
    pd = np.diff(phases, axis=0); pd = np.arctan2(np.sin(pd), np.cos(pd))
    scores = np.array([max(0.0, 1.0-np.std(pd[:,d])/1.5) for d in range(D)])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Phase Structure - {config['name']}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    im = ax.imshow(phases.T, aspect="auto", cmap="hsv", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    ax.set_xlabel("Token ID"); ax.set_ylabel("Dimension"); ax.set_title("(a) Phase Map")
    plt.colorbar(im, ax=ax, label="Phase (rad)", shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(pd.T, aspect="auto", cmap="RdBu_r", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
    ax.set_xlabel("Token Pair"); ax.set_ylabel("Dimension"); ax.set_title("(b) Phase Diff Map")
    plt.colorbar(im, ax=ax, label="dphi (rad)", shrink=0.8)

    ax = axes[1, 0]
    bars = ax.bar(range(D), scores, color="steelblue", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold 0.5")
    nhi = int(np.sum(scores > 0.5))
    ax.set_xlabel("Dimension"); ax.set_ylabel("Linearity")
    ax.set_title(f"(c) Phase Linearity (#High: {nhi}/{D})"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    bd = np.argmax(scores); bars[bd].set_color("orangered"); bars[bd].set_edgecolor("black")

    ax = axes[1, 1]
    uw = np.unwrap(phases[:, bd])
    ax.plot(range(V), uw, color="darkorange", linewidth=0.8, label=f"Dim {bd} ({scores[bd]:.3f})")
    sd = np.argsort(scores)[::-1]
    if D > 1:
        sb = sd[1]; uw2 = np.unwrap(phases[:, sb])
        ax.plot(range(V), uw2, color="royalblue", linewidth=0.6, alpha=0.7, label=f"Dim {sb} ({scores[sb]:.3f})")
    ax.set_xlabel("Token ID"); ax.set_ylabel("Unwrapped Phase"); ax.set_title("(d) Phase Unwrap")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Phase structure: {save_path}")

# =============================================================================
#  Fig 5: Ring Topology (4 panels)
# =============================================================================
def plot_ring_topology(z, config, save_path):
    V, D = z.shape
    dm = (np.abs(z[:, np.newaxis, :] - z[np.newaxis, :, :])**2).sum(axis=-1)
    np.fill_diagonal(dm, np.inf)
    nn_ids = np.argmin(dm, axis=1); tid = np.arange(V)
    nn_rd = np.minimum(np.abs(nn_ids-tid), V-np.abs(nn_ids-tid))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Ring Topology - {config['name']}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]; mrd = min(20, V//2)
    counts, _, patches = ax.hist(nn_rd, bins=np.arange(-0.5, mrd+1.5, 1), color="steelblue", edgecolor="black", alpha=0.8)
    if len(patches) > 1: patches[1].set_facecolor("limegreen")
    nnr = np.mean(nn_rd == 1)
    ax.set_xlabel("Ring Dist to NN"); ax.set_ylabel("Count")
    ax.set_title(f"(a) NN Distribution (NN%={nnr:.1%})"); ax.set_xlim(-0.5, mrd+0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(0.95, 0.95, f"d=1: {int(np.sum(nn_rd==1))}/{V}\nd=2: {int(np.sum(nn_rd==2))}/{V}\nd>=3: {int(np.sum(nn_rd>=3))}/{V}", transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax = axes[0, 1]
    rng = np.random.RandomState(42); pairs = set()
    while len(pairs) < min(5000, V*(V-1)//2):
        i, j = rng.randint(0, V), rng.randint(0, V)
        if i != j: pairs.add((min(i,j), max(i,j)))
    rds, eds = [], []
    for i, j in pairs:
        rds.append(min(abs(i-j), V-abs(i-j)))
        eds.append(np.sqrt(dm[i,j]) if dm[i,j] != np.inf else 0)
    ax.scatter(rds, eds, s=1, alpha=0.3, color="royalblue")
    ax.set_xlabel("Ring Distance"); ax.set_ylabel("Embedding Dist")
    ax.set_title("(b) Ring vs Embedding Dist"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    cc = ["limegreen" if d==1 else "red" for d in nn_rd]
    ax.scatter(range(V), nn_rd, c=cc, s=5, alpha=0.7)
    ax.set_xlabel("Token ID"); ax.set_ylabel("Ring Dist to NN")
    ax.set_title("(c) Per-Token NN Correctness"); ax.set_ylim(-0.5, mrd+0.5); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    nranks = []
    for k in range(V):
        si = np.argsort(dm[k]); rm = {idx: r for r, idx in enumerate(si)}
        nranks.append(rm[(k-1)%V]); nranks.append(rm[(k+1)%V])
    nranks = np.array(nranks)
    ax.hist(nranks, bins=50, color="mediumpurple", edgecolor="black", alpha=0.8)
    med = np.median(nranks)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Rank 0")
    ax.axvline(x=med, color="orange", linestyle="--", alpha=0.5, label=f"Median: {med:.0f}")
    ax.set_xlabel("Rank of Ring Neighbor"); ax.set_ylabel("Count")
    ax.set_title(f"(d) Neighbor Rank (median={med:.0f})"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Ring topology: {save_path}")

# =============================================================================
#  Fig 6: Distance Matrix (3 panels)
# =============================================================================
def plot_distance_matrix(z, config, save_path):
    V, D = z.shape
    diff = z[:, np.newaxis, :] - z[np.newaxis, :, :]
    dm = np.sqrt((np.abs(diff)**2).sum(axis=-1))
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Distance Matrix - {config['name']}", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(dm, cmap="viridis", aspect="equal", interpolation="nearest")
    ax1.set_xlabel("Token ID"); ax1.set_ylabel("Token ID"); ax1.set_title(f"(a) Full ({V}x{V})")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(1, 3, 2); zm = min(50, V)
    im2 = ax2.imshow(dm[:zm,:zm], cmap="viridis", aspect="equal", interpolation="nearest")
    ax2.set_xlabel("Token ID"); ax2.set_ylabel("Token ID"); ax2.set_title(f"(b) Zoomed ({zm}x{zm})")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(1, 3, 3)
    ut = dm[np.triu_indices(V, k=1)]
    ax3.hist(ut, bins=100, color="teal", edgecolor="none", alpha=0.8)
    ax3.axvline(x=np.median(ut), color="red", linestyle="--", label=f"Median: {np.median(ut):.3f}")
    ax3.axvline(x=np.min(ut), color="orange", linestyle=":", label=f"Min: {np.min(ut):.3f}")
    ax3.set_xlabel("L2 Distance"); ax3.set_ylabel("Count"); ax3.set_title("(c) Pairwise Distribution")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Distance matrix: {save_path}")

# =============================================================================
#  Fig 7: Energy Landscape (4 panels)
# =============================================================================
def plot_energy_landscape(z, config, save_path):
    V, D = z.shape; eps = 1e-8
    ze, ee = z[:, np.newaxis, :], z[np.newaxis, :, :]
    r, rh = np.abs(ze)+eps, np.abs(ee)+eps; u, uh = ze/r, ee/rh
    lr = np.log(r) - np.log(rh)
    E_all = (lr**2 + np.abs(u-uh)**2).sum(axis=-1)
    E_tgt = np.diag(E_all); Em = E_all.copy(); np.fill_diagonal(Em, np.inf)
    E_nw = Em.min(axis=1); gap = E_nw - E_tgt
    tau = np.std(E_all, axis=1)+eps; logits = -E_all/tau[:, np.newaxis]
    logits -= logits.max(axis=1, keepdims=True)
    el = np.exp(logits); probs = el/el.sum(axis=1, keepdims=True)
    ptgt = np.array([probs[k,k] for k in range(V)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Energy Landscape - {config['name']}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(E_tgt, E_nw, s=5, alpha=0.6, color="royalblue")
    lim = max(E_nw.max(), E_tgt.max())*1.1
    ax.plot([0,lim], [0,lim], "r--", alpha=0.5, label="E_nw=E_tgt")
    ax.set_xlabel("E_target"); ax.set_ylabel("E_nearest_wrong"); ax.set_title("(a) Target vs Nearest Wrong")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(gap, bins=80, color="seagreen", edgecolor="none", alpha=0.8)
    nc = int(np.sum(gap < 0))
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label=f"Zero (neg: {nc}/{V}={nc/V:.1%})")
    ax.axvline(x=np.median(gap), color="orange", linestyle="--", alpha=0.5, label=f"Median: {np.median(gap):.4f}")
    ax.set_xlabel("E_nw-E_tgt"); ax.set_ylabel("Count"); ax.set_title("(b) Energy Gap Distribution")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    gc = ["red" if g<0 else "seagreen" for g in gap]
    ax.bar(range(V), gap, color=gc, width=1.0, alpha=0.7); ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Token ID"); ax.set_ylabel("Energy Gap"); ax.set_title("(c) Per-Token Gap")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.hist(ptgt, bins=50, color="mediumpurple", edgecolor="none", alpha=0.8)
    ax.axvline(x=1/V, color="red", linestyle="--", alpha=0.5, label=f"Random: {1/V:.4f}")
    ax.axvline(x=np.mean(ptgt), color="orange", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(ptgt):.3f}")
    ax.set_xlabel("p_target"); ax.set_ylabel("Count"); ax.set_title("(d) Self-Boltzmann p_target")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Energy landscape: {save_path}")

# =============================================================================
#  Fig 8: Magnitude Analysis (4 panels)
# =============================================================================
def plot_magnitude_analysis(z, config, save_path):
    V, D = z.shape; mags = np.abs(z)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Magnitude Analysis - {config['name']}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]; trms = np.sqrt(np.mean(mags**2, axis=1))
    ax.plot(range(V), trms, color="teal", linewidth=0.5, alpha=0.8)
    ax.axhline(y=np.mean(trms), color="red", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(trms):.3f}")
    ax.fill_between(range(V), np.mean(trms)-np.std(trms), np.mean(trms)+np.std(trms), alpha=0.15, color="teal", label=f"+/-1s: {np.std(trms):.3f}")
    ax.set_xlabel("Token ID"); ax.set_ylabel("RMS"); ax.set_title("(a) Per-Token RMS")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]; drms = np.sqrt(np.mean(mags**2, axis=0))
    ax.bar(range(D), drms, color="steelblue", alpha=0.8)
    ax.axhline(y=np.mean(drms), color="red", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(drms):.3f}")
    ax.set_xlabel("Dimension"); ax.set_ylabel("RMS"); ax.set_title("(b) Per-Dim RMS")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]; bd = np.argmax(np.var(mags, axis=0))
    sc = ax.scatter(np.angle(z[:,bd]), mags[:,bd], c=np.arange(V), cmap="hsv", s=5, alpha=0.7)
    ax.set_xlabel(f"Phase (dim {bd})"); ax.set_ylabel(f"|z| (dim {bd})")
    ax.set_title(f"(c) Phase-Mag Joint (dim {bd})"); plt.colorbar(sc, ax=ax, label="Token ID", shrink=0.8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    im = ax.imshow(mags.T, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_xlabel("Token ID"); ax.set_ylabel("Dimension"); ax.set_title("(d) Magnitude Heatmap")
    plt.colorbar(im, ax=ax, label="|z|", shrink=0.8)
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Magnitude: {save_path}")

# =============================================================================
#  Fig 9: TDA Persistence (4 panels)
# =============================================================================
def plot_tda_persistence(z, config, save_path):
    try:
        from ripser import ripser
    except ImportError:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "ripser not installed\npip install ripser", transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray", bbox=dict(boxstyle="round", facecolor="lightyellow"))
        ax.set_title("TDA - SKIPPED"); ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        print(f"  [SKIP] TDA: ripser not installed"); return

    V, D = z.shape; z_real = np.column_stack([z.real, z.imag])
    print("  [TDA] Computing persistent homology (10-60s)...")
    res = ripser(z_real, maxdim=1); h0, h1 = res["dgms"][0], res["dgms"][1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"TDA Persistence - {config['name']}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if len(h0) > 0:
        fh = h0[np.isfinite(h0[:,1])]; ih = h0[~np.isfinite(h0[:,1])]
        if len(fh) > 0: ax.scatter(fh[:,0], fh[:,1], s=10, alpha=0.5, color="steelblue")
        if len(ih) > 0:
            md = fh[:,1].max() if len(fh)>0 else 1
            ax.scatter(ih[:,0], [md*1.1]*len(ih), s=30, marker="^", color="red", label="Inf")
        dmax = max(h0[np.isfinite(h0[:,1]),1].max(), 1) if np.any(np.isfinite(h0[:,1])) else 1
        ax.plot([0,dmax], [0,dmax], "k--", alpha=0.3)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death"); ax.set_title(f"(a) H0 ({len(h0)})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if len(h1) > 0:
        ps = h1[:,1]-h1[:,0]
        sc = ax.scatter(h1[:,0], h1[:,1], c=ps, cmap="hot_r", s=15, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Persistence", shrink=0.8)
        ax.plot([0,h1[:,1].max()], [0,h1[:,1].max()], "k--", alpha=0.3)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death"); ax.set_title(f"(b) H1 ({len(h1)})"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(h1) > 0:
        sh = h1[np.argsort(h1[:,1]-h1[:,0])[::-1]]; ns = min(30, len(sh))
        for i in range(ns):
            ax.plot([sh[i,0],sh[i,1]], [i,i], color="orangered" if i==0 else "steelblue", linewidth=3 if i==0 else 1, alpha=0.8)
    ax.set_xlabel("Filtration"); ax.set_ylabel("Feature"); ax.set_title(f"(c) H1 Barcode (top {min(30,len(h1))})")
    ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()

    ax = axes[1, 1]
    if len(h1) > 0:
        pss = sorted(h1[:,1]-h1[:,0], reverse=True)
        ax.bar(range(len(pss)), pss, color="mediumpurple", alpha=0.8)
        if len(pss) >= 2:
            ax.text(0.95, 0.95, f"Top: {pss[0]:.4f}\n2nd: {pss[1]:.4f}\nRatio: {pss[0]/pss[1]:.2f}", transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_xlabel("Rank"); ax.set_ylabel("Persistence"); ax.set_title("(d) H1 Ranking"); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] TDA: {save_path}")

# =============================================================================
#  Fig 10: Attention Pattern
# =============================================================================
def _save_placeholder(save_path, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="gray")
    ax.set_title(title); ax.axis("off"); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)

def plot_attention_pattern(data_dir, config, save_path, device="cpu"):
    try:
        import torch
        from Anla.experiments.capacity.capacity_pressure_test_v4 import AnlaManifoldInpainter
    except ImportError:
        _save_placeholder(save_path, "Attention - Import Failed"); return
    checkpoint = load_checkpoint(data_dir, device)
    if checkpoint is None: _save_placeholder(save_path, "Attention - No Ckpt"); return
    model = reconstruct_model(checkpoint, device)
    if model is None: _save_placeholder(save_path, "Attention - Model Failed"); return
    import torch
    cfg = checkpoint["config"]; vs, sl = cfg["vocab_size"], cfg["seq_len"]
    seq = [(0+i)%vs for i in range(sl)]
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    mpos = [15, 16, 17]
    for p in mpos: ids[0, p] = vs
    model.train()
    with torch.no_grad(): model.forward(ids)
    aw = model.block.attn.attn_cache
    if aw is None: _save_placeholder(save_path, "Attention - Cache Empty"); model.eval(); return
    model.eval()
    am = aw.abs().cpu().numpy()[0]; ap = torch.angle(aw).cpu().numpy()[0]; nh = am.shape[0]
    fig, axes = plt.subplots(2, nh, figsize=(5*nh, 9))
    if nh == 1: axes = axes.reshape(2, 1)
    fig.suptitle(f"Attention - {config['name']} (MASK@{mpos})", fontsize=14, fontweight="bold")
    for h in range(nh):
        ax = axes[0, h]
        im = ax.imshow(am[h], cmap="Blues", aspect="equal", interpolation="nearest", vmin=0)
        ax.set_xlabel("Key"); ax.set_ylabel("Query"); ax.set_title(f"Head {h} - |A|")
        plt.colorbar(im, ax=ax, shrink=0.7)
        for p in mpos: ax.axhline(y=p, color="red", linewidth=0.5, alpha=0.5)
        ax = axes[1, h]
        im = ax.imshow(ap[h], cmap="hsv", aspect="equal", interpolation="nearest", vmin=-np.pi, vmax=np.pi)
        ax.set_xlabel("Key"); ax.set_ylabel("Query"); ax.set_title(f"Head {h} - Phase")
        plt.colorbar(im, ax=ax, shrink=0.7, label="rad")
        for p in mpos: ax.axhline(y=p, color="red", linewidth=0.5, alpha=0.5)
    plt.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print(f"  [OK] Attention: {save_path}")

# =============================================================================
#  Summary Report
# =============================================================================
def generate_summary_report(result, z, save_path):
    cfg = result["config"]; V, D = z.shape; mags = np.abs(z)
    phases = np.angle(z); pd = np.diff(phases, axis=0); pd = np.arctan2(np.sin(pd), np.cos(pd))
    diff = z[:, np.newaxis, :]-z[np.newaxis, :, :]
    dm = np.sqrt((np.abs(diff)**2).sum(axis=-1))
    np.fill_diagonal(dm, np.inf); nn_ids = np.argmin(dm, axis=1)
    nn_rd = np.minimum(np.abs(nn_ids-np.arange(V)), V-np.abs(nn_ids-np.arange(V)))
    pds = np.array([max(0.0, 1.0-np.std(pd[:,d])/1.5) for d in range(D)])

    L = []
    L.append("="*80); L.append("  Config B - Quantitative Summary"); L.append("="*80)
    L.append(f"\n  I. Config: V={cfg['vocab_size']}, D={cfg['d_model']}, V/D={cfg['vocab_size']//cfg['d_model']}")
    L.append(f"     Heads={cfg['num_heads']}, HeadDim={cfg['d_model']//cfg['num_heads']}, FFN={cfg['d_model']*4}")
    tp = result.get('total_params', 'N/A')
    L.append(f"     Params: {tp:,}" if isinstance(tp, int) else f"     Params: {tp}")
    bta = result.get('best_train_acc', 0); bte_ep = result.get('best_train_epoch', '?')
    L.append(f"\n  II. Results")
    L.append(f"     Train Acc:  {bta:.2%} (epoch {bte_ep})")
    fta = result.get('final_test_acc', 0); fts = result.get('final_test_std', 0)
    L.append(f"     Test Acc:   {fta:.2%} +/- {fts:.2%}")
    L.append(f"     Time:       {result.get('training_time_sec',0):.0f}s")
    L.append(f"     rho:        {result.get('final_rho',0):.4f}")
    L.append(f"\n  III. Boltzmann")
    L.append(f"     tau:        {result.get('final_tau','N/A')}")
    L.append(f"     p_tgt:      {result.get('final_p_target_mean','N/A')}")
    L.append(f"     neg%:       {result.get('final_negative_margin_ratio','N/A')}")
    L.append(f"\n  IV. Geometry")
    L.append(f"     Mag mean/std: {np.mean(mags):.4f}/{np.std(mags):.4f}")
    welch = np.sqrt((V-D)/(D*(V-1)))
    L.append(f"     Welch LB:   {welch:.4f}")
    L.append(f"     JL UB:      {np.sqrt(np.log(V)/(2*D)):.4f}")
    L.append(f"\n  V. Ring Topology")
    L.append(f"     NN%:        {np.mean(nn_rd==1):.1%} ({np.sum(nn_rd==1)}/{V})")
    ut = dm[np.triu_indices(V, k=1)]; vut = ut[ut < np.inf]
    L.append(f"     Dist min/med/max: {np.min(vut):.4f}/{np.median(vut):.4f}/{np.max(vut):.4f}")
    L.append(f"\n  VI. Phase")
    L.append(f"     Linearity max/mean: {np.max(pds):.4f}/{np.mean(pds):.4f}")
    L.append(f"     #High(>0.5): {np.sum(pds>0.5)}/{D}")
    tda = result.get("tda_quick", {})
    if tda.get("available"):
        L.append(f"\n  VII. TDA: H1={tda.get('h1_count','?')}, dom={tda.get('dominant_persistence',0):.4f}")
    else:
        L.append(f"\n  VII. TDA: unavailable")
    alpha = (V-1)/(2*D)
    L.append(f"\n  VIII. Theory: alpha={alpha:.2f}, freq_cov={D}/{V//2}={D/(V//2):.1%}")
    L.append("\n"+"="*80)
    report = "\n".join(L)
    with open(save_path, "w", encoding="utf-8") as f: f.write(report)
    print(report); print(f"\n  [OK] Summary: {save_path}")

# =============================================================================
#  Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Config B Visualization")
    parser.add_argument("--data-dir", type=str, default=os.path.join(_ANLA_ROOT, "Logs", "config_B_analysis", "config_B_v256_d64"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    data_dir = args.data_dir; vis_dir = os.path.join(data_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    print("="*72); print("  Config B Full Visualization"); print("="*72)
    print(f"  Data: {data_dir}\n  Output: {vis_dir}\n")
    result = load_training_log(data_dir); history, config = result["history"], result["config"]
    z = load_embedding(data_dir); print(f"  Embedding: {z.shape}, {z.dtype}\n")
    plot_training_dynamics(history, config, result, os.path.join(vis_dir, "01_training_dynamics.png"))
    plot_boltzmann_diagnostics(history, config, os.path.join(vis_dir, "02_boltzmann_diagnostics.png"))
    plot_embedding_pca(z, config, os.path.join(vis_dir, "03_embedding_pca.png"))
    plot_phase_structure(z, config, os.path.join(vis_dir, "04_phase_structure.png"))
    plot_ring_topology(z, config, os.path.join(vis_dir, "05_ring_topology.png"))
    plot_distance_matrix(z, config, os.path.join(vis_dir, "06_distance_matrix.png"))
    plot_energy_landscape(z, config, os.path.join(vis_dir, "07_energy_landscape.png"))
    plot_magnitude_analysis(z, config, os.path.join(vis_dir, "08_magnitude_analysis.png"))
    plot_tda_persistence(z, config, os.path.join(vis_dir, "09_tda_persistence.png"))
    plot_attention_pattern(data_dir, config, os.path.join(vis_dir, "10_attention_pattern.png"), device=args.device)
    generate_summary_report(result, z, os.path.join(vis_dir, "summary_report.txt"))
    print(f"\n{'='*72}\n  Done! Output: {vis_dir}\n{'='*72}")

if __name__ == "__main__":
    main()
