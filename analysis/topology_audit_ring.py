import argparse
import json
import os
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

try:
    from ripser import ripser
    from persim import plot_diagrams
except ImportError as exc:
    raise ImportError(
        "Missing dependency. Please run: pip install ripser persim"
    ) from exc

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

def _to_builtin(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _find_embedding_key(state_dict: Dict[str, torch.Tensor]) -> str:
    preferred = [
        "embedding.weight",
        "embed.weight",
        "model.embedding.weight",
        "model.embed.weight",
    ]
    for key in preferred:
        if key in state_dict:
            return key

    for key in state_dict.keys():
        if key.endswith("embedding.weight") or key.endswith("embed.weight"):
            return key

    raise KeyError("Cannot find embedding weight key in checkpoint.")


def _load_embedding_from_checkpoint(checkpoint_path: str, vocab_size: int) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    key = _find_embedding_key(state_dict)
    weight = state_dict[key].detach().cpu().numpy()

    if weight.shape[0] < vocab_size:
        raise ValueError(
            f"Checkpoint embedding rows={weight.shape[0]} < vocab_size={vocab_size}"
        )

    weight = weight[:vocab_size]
    if not np.iscomplexobj(weight):
        raise ValueError(
            f"Embedding is not complex dtype (dtype={weight.dtype})."
        )
    return weight


def _complex_to_real_points(weight_complex: np.ndarray) -> np.ndarray:
    return np.concatenate([weight_complex.real, weight_complex.imag], axis=1).astype(np.float64)


def _pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    return squareform(pdist(points, metric="euclidean"))


def _build_geodesic_distance_matrix(
    points: np.ndarray,
    initial_k: int = 6
) -> Tuple[np.ndarray, int]:
    """
    Build kNN graph geodesic distance matrix.
    Auto-increases k until graph becomes connected.
    """
    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for topology audit.")

    start_k = max(2, initial_k)
    for k in range(start_k, n):
        n_neighbors = min(k + 1, n)  # +1 because self is included
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)

        # remove self column
        neighbor_count = indices.shape[1] - 1
        rows = np.repeat(np.arange(n), neighbor_count)
        cols = indices[:, 1:].reshape(-1)
        vals = distances[:, 1:].reshape(-1)

        graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
        graph = graph.minimum(graph.T)

        n_components, _ = connected_components(graph, directed=False)
        if n_components == 1:
            d_geo = shortest_path(graph, directed=False, unweighted=False)
            return d_geo, k

    raise RuntimeError("kNN graph could not be connected even with large k.")


def _distance_scale(distance_matrix: np.ndarray) -> float:
    tri = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    if tri.size == 0:
        return 1.0
    return float(np.median(tri))


def _h1_stats(
    diagrams,
    distance_matrix: np.ndarray,
    persistence_ratio: float = 0.10
) -> Dict[str, float]:
    """
    Extract robust H1 stats:
      - dominant persistence
      - normalized persistence
      - significant feature count
      - dominance ratio over second largest feature
    """
    if len(diagrams) < 2:
        return {
            "h1_count_total": 0,
            "h1_count_finite": 0,
            "h1_significant_count": 0,
            "h1_dominant_persistence": 0.0,
            "h1_second_persistence": 0.0,
            "h1_dominance_ratio": 0.0,
            "h1_dominant_persistence_norm": 0.0,
            "h1_significant_threshold": 0.0,
            "distance_scale": _distance_scale(distance_matrix),
            "h1_betti_at_dominant_mid": 0,
        }

    dgm_h1 = diagrams[1]
    scale = _distance_scale(distance_matrix)
    threshold = persistence_ratio * scale

    if dgm_h1.shape[0] == 0:
        return {
            "h1_count_total": 0,
            "h1_count_finite": 0,
            "h1_significant_count": 0,
            "h1_dominant_persistence": 0.0,
            "h1_second_persistence": 0.0,
            "h1_dominance_ratio": 0.0,
            "h1_dominant_persistence_norm": 0.0,
            "h1_significant_threshold": threshold,
            "distance_scale": scale,
            "h1_betti_at_dominant_mid": 0,
        }

    finite_mask = np.isfinite(dgm_h1[:, 1])
    finite = dgm_h1[finite_mask]

    if finite.shape[0] == 0:
        return {
            "h1_count_total": int(dgm_h1.shape[0]),
            "h1_count_finite": 0,
            "h1_significant_count": 0,
            "h1_dominant_persistence": 0.0,
            "h1_second_persistence": 0.0,
            "h1_dominance_ratio": 0.0,
            "h1_dominant_persistence_norm": 0.0,
            "h1_significant_threshold": threshold,
            "distance_scale": scale,
            "h1_betti_at_dominant_mid": 0,
        }

    lifetimes = finite[:, 1] - finite[:, 0]
    order = np.argsort(-lifetimes)
    sorted_lt = lifetimes[order]

    dominant = float(sorted_lt[0])
    second = float(sorted_lt[1]) if sorted_lt.shape[0] > 1 else 0.0
    dominance_ratio = dominant / (second + 1e-12) if dominant > 0 else 0.0
    dominant_norm = dominant / (scale + 1e-12)

    significant_count = int(np.sum(lifetimes >= threshold))

    # Betti-1 at midpoint of dominant bar
    dom_idx = order[0]
    b, d = finite[dom_idx]
    eps_mid = float(b + 0.5 * (d - b))
    betti1_mid = int(np.sum((dgm_h1[:, 0] <= eps_mid) & (eps_mid < dgm_h1[:, 1])))

    return {
        "h1_count_total": int(dgm_h1.shape[0]),
        "h1_count_finite": int(finite.shape[0]),
        "h1_significant_count": significant_count,
        "h1_dominant_persistence": dominant,
        "h1_second_persistence": second,
        "h1_dominance_ratio": dominance_ratio,
        "h1_dominant_persistence_norm": dominant_norm,
        "h1_significant_threshold": threshold,
        "distance_scale": scale,
        "h1_betti_at_dominant_mid": betti1_mid,
    }


def _plot_h1_barcode(ax, dgm_h1: np.ndarray, threshold: float, title: str, max_bars: int = 30):
    ax.set_title(title)
    if dgm_h1.shape[0] == 0:
        ax.text(0.5, 0.5, "No H1 features", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Filtration")
        ax.set_ylabel("Bar index")
        return

    finite = dgm_h1[np.isfinite(dgm_h1[:, 1])]
    if finite.shape[0] == 0:
        ax.text(0.5, 0.5, "No finite H1 bars", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Filtration")
        ax.set_ylabel("Bar index")
        return

    lifetimes = finite[:, 1] - finite[:, 0]
    order = np.argsort(-lifetimes)
    finite = finite[order][:max_bars]
    lifetimes = lifetimes[order][:max_bars]

    for i, (bar, lt) in enumerate(zip(finite, lifetimes)):
        b, d = bar
        color = "#dc2626" if lt >= threshold else "#9ca3af"
        ax.hlines(i, b, d, colors=color, linewidth=2.2)

    ax.set_xlabel("Filtration")
    ax.set_ylabel("Bar index")
    ax.grid(alpha=0.2)


def _verdict(geo_stats: Dict[str, float]) -> str:
    sig = geo_stats["h1_significant_count"]
    dom_norm = geo_stats["h1_dominant_persistence_norm"]
    dom_ratio = geo_stats["h1_dominance_ratio"]

    # empirically practical thresholds for small N=64 point-clouds
    if sig >= 1 and dom_norm >= 0.12 and dom_ratio >= 1.5:
        return "Strong evidence of one dominant loop (S1), possibly twisted in high dimension."
    if sig >= 1 and dom_norm >= 0.08:
        return "Loop signal exists, but may be noisy or mixed with extra local cycles."
    return "No robust loop evidence under geodesic metric."


def run_topology_audit_from_embedding(
    embedding_complex: torch.Tensor,
    output_dir: str = os.path.join("Logs", "ring_masking_vis"),
    initial_k: int = 6,
    persistence_ratio: float = 0.10,
    maxdim: int = 2
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(embedding_complex, torch.Tensor):
        weight_complex = embedding_complex.detach().cpu().numpy()
    else:
        weight_complex = np.asarray(embedding_complex)

    if weight_complex.ndim != 2:
        raise ValueError(f"Expected shape [V, D], got {weight_complex.shape}")
    if not np.iscomplexobj(weight_complex):
        raise ValueError("Embedding must be complex-valued for this audit.")

    points = _complex_to_real_points(weight_complex)

    # Euclidean metric PH
    d_euc = _pairwise_distance_matrix(points)
    dgms_euc = ripser(d_euc, distance_matrix=True, maxdim=maxdim)["dgms"]
    stats_euc = _h1_stats(dgms_euc, d_euc, persistence_ratio=persistence_ratio)

    # Geodesic metric PH (kNN shortest path)
    d_geo, used_k = _build_geodesic_distance_matrix(points, initial_k=initial_k)
    dgms_geo = ripser(d_geo, distance_matrix=True, maxdim=maxdim)["dgms"]
    stats_geo = _h1_stats(dgms_geo, d_geo, persistence_ratio=persistence_ratio)

    verdict = _verdict(stats_geo)

    # Plot report
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_diagrams(dgms_euc, ax=axes[0, 0], show=False)
    axes[0, 0].set_title("Persistence Diagram (Euclidean)")

    plot_diagrams(dgms_geo, ax=axes[0, 1], show=False)
    axes[0, 1].set_title(f"Persistence Diagram (Geodesic kNN, k={used_k})")

    dgm_h1_euc = dgms_euc[1] if len(dgms_euc) > 1 else np.empty((0, 2))
    dgm_h1_geo = dgms_geo[1] if len(dgms_geo) > 1 else np.empty((0, 2))
    _plot_h1_barcode(
        axes[1, 0],
        dgm_h1_euc,
        threshold=stats_euc["h1_significant_threshold"],
        title="H1 Barcode (Euclidean)"
    )
    _plot_h1_barcode(
        axes[1, 1],
        dgm_h1_geo,
        threshold=stats_geo["h1_significant_threshold"],
        title="H1 Barcode (Geodesic)"
    )

    fig.suptitle("Ring Topology Audit via Persistent Homology", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(output_dir, "TDA_REPORT.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    report = {
        "knn_k_used": used_k,
        "persistence_ratio": persistence_ratio,
        "euclidean_h1": stats_euc,
        "geodesic_h1": stats_geo,
        "verdict": verdict,
        "report_figure": fig_path,
    }

    json_path = os.path.join(output_dir, "TDA_SUMMARY.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(report), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("Route-1 Topology Audit (Persistent Homology)")
    print("=" * 64)
    print(f"kNN geodesic k: {used_k}")
    print(f"Euclidean H1 significant count: {stats_euc['h1_significant_count']}")
    print(f"Euclidean dominant persistence(norm): {stats_euc['h1_dominant_persistence_norm']:.4f}")
    print(f"Geodesic H1 significant count: {stats_geo['h1_significant_count']}")
    print(f"Geodesic dominant persistence(norm): {stats_geo['h1_dominant_persistence_norm']:.4f}")
    print(f"Geodesic dominance ratio: {stats_geo['h1_dominance_ratio']:.3f}")
    print(f"Verdict: {verdict}")
    print(f"Figure: {fig_path}")
    print(f"JSON : {json_path}")
    print("=" * 64 + "\n")

    return report


def run_topology_audit_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int = 64,
    output_dir: str = os.path.join("Logs", "ring_masking_vis"),
    initial_k: int = 6,
    persistence_ratio: float = 0.10,
    maxdim: int = 2
):
    weight_complex = _load_embedding_from_checkpoint(checkpoint_path, vocab_size=vocab_size)
    return run_topology_audit_from_embedding(
        weight_complex,
        output_dir=output_dir,
        initial_k=initial_k,
        persistence_ratio=persistence_ratio,
        maxdim=maxdim
    )


def main():
    parser = argparse.ArgumentParser(description="Route-1 topology audit for ring embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=os.path.join("Logs", "ring_masking_vis"))
    parser.add_argument("--initial_k", type=int, default=6)
    parser.add_argument("--persistence_ratio", type=float, default=0.10)
    parser.add_argument("--maxdim", type=int, default=2)
    args = parser.parse_args()

    run_topology_audit_from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        initial_k=args.initial_k,
        persistence_ratio=args.persistence_ratio,
        maxdim=args.maxdim
    )


if __name__ == "__main__":
    main()
