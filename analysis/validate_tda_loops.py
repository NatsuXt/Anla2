import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from ripser import ripser


def ensure_complex_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        z = arr
    elif arr.ndim >= 1 and arr.shape[-1] == 2:
        z = arr[..., 0] + 1j * arr[..., 1]
    else:
        raise ValueError(
            "输入不是复数矩阵，也不是 [...,2] 的实虚拼接格式。"
        )

    if z.ndim == 1:
        z = z[:, None]
    elif z.ndim > 2:
        z = z.reshape(-1, z.shape[-1])

    return z.astype(np.complex128, copy=False)


def extract_from_dict(obj: Dict[str, Any], key: str = None) -> Any:
    if key is not None and key in obj:
        return obj[key]

    for cand in ["z", "features", "embeddings", "latent", "points"]:
        if cand in obj:
            return obj[cand]

    if len(obj) == 1:
        return next(iter(obj.values()))

    raise ValueError(
        f"无法从 dict 中自动提取特征，现有 keys={list(obj.keys())}，请用 --key 指定。"
    )


def load_features(path: str, key: str = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            obj = arr.item()
            if isinstance(obj, dict):
                arr = extract_from_dict(obj, key)
        return ensure_complex_matrix(arr)

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"{path} 中没有可用键。")
            if key is None:
                chosen = keys[0]
            else:
                if key not in data:
                    raise ValueError(f"--key={key} 不在 {keys} 中。")
                chosen = key
            arr = data[chosen]
        return ensure_complex_matrix(arr)

    if ext in [".pt", ".pth"]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("读取 .pt/.pth 需要安装 torch。") from exc

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            arr = extract_from_dict(obj, key)
        else:
            arr = obj

        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        return ensure_complex_matrix(arr)

    raise ValueError(f"不支持的文件后缀: {ext}")


def pairwise_euclidean(z: np.ndarray) -> np.ndarray:
    x = np.concatenate([z.real, z.imag], axis=1).astype(np.float64, copy=False)
    return squareform(pdist(x, metric="euclidean"))


def knn_geodesic(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[0]
    if n < 3:
        raise ValueError("样本点太少，无法构造稳定 geodesic 图。")

    k = max(1, min(k, n - 1))
    graph = lil_matrix((n, n), dtype=np.float64)

    for i in range(n):
        idx = np.argpartition(D[i], k + 1)[: k + 1]
        idx = idx[idx != i]
        for j in idx:
            graph[i, j] = D[i, j]

    graph = graph.tocsr()
    graph = graph.maximum(graph.T)

    Dg = shortest_path(graph, directed=False, unweighted=False)
    finite = np.isfinite(Dg)

    if not finite.all():
        if np.any(finite):
            max_finite = float(np.max(Dg[finite]))
        else:
            max_finite = 1.0
        Dg[~finite] = max_finite * 1.05

    return Dg


def h1_persistence(D: np.ndarray) -> np.ndarray:
    result = ripser(D, distance_matrix=True, maxdim=1)
    dgm1 = result["dgms"][1]

    if dgm1.size == 0:
        return np.empty(0, dtype=np.float64)

    mask = np.isfinite(dgm1[:, 0]) & np.isfinite(dgm1[:, 1])
    dgm1 = dgm1[mask]
    if dgm1.size == 0:
        return np.empty(0, dtype=np.float64)

    pers = dgm1[:, 1] - dgm1[:, 0]
    pers = pers[pers > 0]
    if pers.size == 0:
        return np.empty(0, dtype=np.float64)

    return np.sort(pers)[::-1].astype(np.float64)


def distance_scale(D: np.ndarray) -> float:
    finite = D[np.isfinite(D)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(finite, 95))


def summarize_pers(pers: np.ndarray, scale: float, threshold: float) -> Dict[str, Any]:
    dominant = float(pers[0]) if pers.size > 0 else 0.0
    second = float(pers[1]) if pers.size > 1 else 0.0

    if np.isfinite(threshold):
        sig_count = int(np.sum(pers >= threshold))
    else:
        sig_count = 0

    ratio = None
    if second > 0:
        ratio = float(dominant / (second + 1e-12))

    return {
        "h1_total_count": int(pers.size),
        "h1_significant_count": sig_count,
        "dominant_persistence": dominant,
        "dominant_persistence_norm": float(dominant / (scale + 1e-12)),
        "dominance_ratio": ratio,
    }


def observed_metrics(z: np.ndarray, k: int, threshold: float) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    De = pairwise_euclidean(z)
    pe = h1_persistence(De)

    Dg = knn_geodesic(De, k)
    pg = h1_persistence(Dg)

    eu = summarize_pers(pe, distance_scale(De), threshold)
    geo = summarize_pers(pg, distance_scale(Dg), threshold)
    return eu, geo, pe, pg


def geodesic_persistence_only(z: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    De = pairwise_euclidean(z)
    Dg = knn_geodesic(De, k)
    pg = h1_persistence(Dg)
    sc = distance_scale(Dg)
    return pg, sc


def make_null(z: np.ndarray, rng: np.random.Generator, model: str) -> np.ndarray:
    mag = np.abs(z)

    if model == "phase_permute":
        theta = np.angle(z).copy()
        n, d = theta.shape
        for j in range(d):
            theta[:, j] = theta[rng.permutation(n), j]
        return mag * np.exp(1j * theta)

    if model == "phase_uniform":
        theta = rng.uniform(-np.pi, np.pi, size=z.shape)
        return mag * np.exp(1j * theta)

    if model == "matched_noise":
        real = rng.normal(
            loc=np.mean(z.real, axis=0, keepdims=True),
            scale=np.std(z.real, axis=0, keepdims=True) + 1e-8,
            size=z.shape,
        )
        imag = rng.normal(
            loc=np.mean(z.imag, axis=0, keepdims=True),
            scale=np.std(z.imag, axis=0, keepdims=True) + 1e-8,
            size=z.shape,
        )
        return real + 1j * imag

    raise ValueError(f"未知 null model: {model}")


def phase_linearity_scores(z: np.ndarray, max_freq: int = 4) -> np.ndarray:
    n, d = z.shape
    t = np.arange(n, dtype=np.float64) / max(1, n)
    theta = np.angle(z)
    scores = np.zeros(d, dtype=np.float64)

    for j in range(d):
        ph = theta[:, j]
        best = 0.0
        for freq in range(1, max_freq + 1):
            ref = 2 * np.pi * freq * t
            score = np.abs(np.mean(np.exp(1j * (ph - ref))))
            if score > best:
                best = score
        scores[j] = best

    return scores


def subsample_stability(
    z: np.ndarray,
    k: int,
    rng: np.random.Generator,
    runs: int,
    frac: float,
    null_model: str,
) -> Dict[str, float]:
    n = z.shape[0]
    m = int(round(n * frac))
    m = max(8, min(m, n))

    obs_dom: List[float] = []
    null_dom: List[float] = []

    for _ in range(runs):
        idx = rng.choice(n, size=m, replace=False)
        z_sub = z[idx]

        pg_obs, _ = geodesic_persistence_only(z_sub, k)
        obs_dom.append(float(pg_obs[0]) if pg_obs.size > 0 else 0.0)

        z_null = make_null(z_sub, rng, null_model)
        pg_null, _ = geodesic_persistence_only(z_null, k)
        null_dom.append(float(pg_null[0]) if pg_null.size > 0 else 0.0)

    obs_dom = np.asarray(obs_dom, dtype=np.float64)
    null_dom = np.asarray(null_dom, dtype=np.float64)

    return {
        "obs_dominant_mean": float(np.mean(obs_dom)),
        "obs_dominant_cv": float(np.std(obs_dom) / (np.mean(obs_dom) + 1e-12)),
        "null_dominant_mean": float(np.mean(null_dom)),
        "null_dominant_cv": float(np.std(null_dom) / (np.mean(null_dom) + 1e-12)),
    }


def maybe_plot(path: str, obs_dom: float, null_dom: np.ndarray, obs_count: int, null_count: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warn] matplotlib 未安装，跳过绘图。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(null_dom, bins=30, color="#4C78A8", alpha=0.85)
    axes[0].axvline(obs_dom, color="#E45756", linestyle="--", linewidth=2)
    axes[0].set_title("Geodesic Dominant Persistence")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("count")

    bins2 = min(30, max(10, len(np.unique(null_count))))
    axes[1].hist(null_count, bins=bins2, color="#72B7B2", alpha=0.85)
    axes[1].axvline(obs_count, color="#E45756", linestyle="--", linewidth=2)
    axes[1].set_title("Significant H1 Count")
    axes[1].set_xlabel("value")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="验证 TDA 的 H1 环是否显著偏离随机噪声。"
    )
    parser.add_argument("--features", type=str, required=True, help="输入特征文件: .npy/.npz/.pt/.pth")
    parser.add_argument("--key", type=str, default=None, help="当输入为 dict/npz 时的键名")
    parser.add_argument("--k", type=int, default=6, help="kNN geodesic 的 k")
    parser.add_argument("--n-null", type=int, default=200, help="零假设重采样次数")
    parser.add_argument(
        "--null-model",
        type=str,
        default="phase_permute",
        choices=["phase_permute", "phase_uniform", "matched_noise"],
        help="零假设模型",
    )
    parser.add_argument("--null-quantile", type=float, default=0.95, help="显著性阈值分位数")
    parser.add_argument("--alpha", type=float, default=0.01, help="显著性检验阈值")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--max-points", type=int, default=1200, help="点太多时随机下采样上限")
    parser.add_argument("--l2-normalize", action="store_true", help="是否对每个点做 L2 归一化")
    parser.add_argument("--dim-slice", type=int, default=0, help="线性度 top/bottom 维度数量，0=自动")
    parser.add_argument("--dim-random-runs", type=int, default=50, help="随机维度对照次数")
    parser.add_argument("--max-freq", type=int, default=4, help="相位线性度评分最大频率")
    parser.add_argument("--subsample-runs", type=int, default=20, help="子采样稳定性重复次数")
    parser.add_argument("--subsample-frac", type=float, default=0.8, help="每次子采样比例")
    parser.add_argument("--out", type=str, default="LOOP_VALIDATION.json", help="输出 JSON")
    parser.add_argument("--plot", type=str, default="", help="可选输出图路径")
    args = parser.parse_args()

    if args.n_null < 20:
        raise ValueError("--n-null 建议 >= 20")
    if not (0.5 <= args.subsample_frac <= 1.0):
        raise ValueError("--subsample-frac 建议在 [0.5, 1.0]")
    if not (0.5 < args.null_quantile < 1.0):
        raise ValueError("--null-quantile 必须在 (0.5, 1.0)")

    rng = np.random.default_rng(args.seed)

    z = load_features(args.features, key=args.key)
    n0, d0 = z.shape
    if n0 < 20:
        raise ValueError(f"样本数太少: {n0}，建议至少 20。")

    if args.l2_normalize:
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / (norms + 1e-12)

    if args.max_points > 0 and z.shape[0] > args.max_points:
        idx = rng.choice(z.shape[0], size=args.max_points, replace=False)
        z = z[idx]

    n, d = z.shape
    if d < 1:
        raise ValueError("特征维度必须 >= 1。")

    eu0, geo0, pe_obs, pg_obs = observed_metrics(z, args.k, threshold=0.0)
    obs_dom = float(geo0["dominant_persistence"])

    null_pers: List[np.ndarray] = []
    null_dom: List[float] = []

    for _ in range(args.n_null):
        z_null = make_null(z, rng, args.null_model)
        pg_null, _ = geodesic_persistence_only(z_null, args.k)
        null_pers.append(pg_null)
        null_dom.append(float(pg_null[0]) if pg_null.size > 0 else 0.0)

    null_dom_arr = np.asarray(null_dom, dtype=np.float64)
    non_empty = [p for p in null_pers if p.size > 0]
    if len(non_empty) > 0:
        all_null_pers = np.concatenate(non_empty, axis=0)
        threshold = float(np.quantile(all_null_pers, args.null_quantile))
    else:
        threshold = np.inf

    obs_sig_count = int(np.sum(pg_obs >= threshold)) if np.isfinite(threshold) else 0
    null_sig_count = np.asarray(
        [int(np.sum(p >= threshold)) if np.isfinite(threshold) else 0 for p in null_pers],
        dtype=np.int32,
    )

    p_value_dom = float((1 + np.sum(null_dom_arr >= obs_dom)) / (len(null_dom_arr) + 1))
    p_value_count = float((1 + np.sum(null_sig_count >= obs_sig_count)) / (len(null_sig_count) + 1))
    z_score_dom = float((obs_dom - np.mean(null_dom_arr)) / (np.std(null_dom_arr) + 1e-12))

    eu, geo, _, _ = observed_metrics(z, args.k, threshold=threshold)

    if d >= 2:
        scores = phase_linearity_scores(z, max_freq=args.max_freq)
        if args.dim_slice <= 0:
            dim_slice = max(2, min(32, max(1, d // 8)))
        else:
            dim_slice = args.dim_slice
        dim_slice = int(max(1, min(dim_slice, d // 2)))

        order = np.argsort(scores)
        bottom_idx = order[:dim_slice]
        top_idx = order[-dim_slice:]

        pg_top, sc_top = geodesic_persistence_only(z[:, top_idx], args.k)
        top_geo = summarize_pers(pg_top, sc_top, threshold)

        pg_bottom, sc_bottom = geodesic_persistence_only(z[:, bottom_idx], args.k)
        bottom_geo = summarize_pers(pg_bottom, sc_bottom, threshold)

        rand_dom = []
        for _ in range(args.dim_random_runs):
            ridx = rng.choice(d, size=dim_slice, replace=False)
            pg_r, _ = geodesic_persistence_only(z[:, ridx], args.k)
            rand_dom.append(float(pg_r[0]) if pg_r.size > 0 else 0.0)

        rand_dom = np.asarray(rand_dom, dtype=np.float64)
        p_top_vs_rand = float((1 + np.sum(rand_dom >= top_geo["dominant_persistence"])) / (len(rand_dom) + 1))

        dim_consistency = {
            "available": True,
            "dim_slice": dim_slice,
            "top_linearity_mean": float(np.mean(scores[top_idx])),
            "bottom_linearity_mean": float(np.mean(scores[bottom_idx])),
            "top_geodesic_dominant": float(top_geo["dominant_persistence"]),
            "bottom_geodesic_dominant": float(bottom_geo["dominant_persistence"]),
            "p_top_vs_random": p_top_vs_rand,
        }
    else:
        dim_consistency = {
            "available": False,
            "reason": "特征维度 < 2，无法做 top/bottom 维度一致性检验。"
        }
        p_top_vs_rand = 1.0

    stability = subsample_stability(
        z=z,
        k=args.k,
        rng=rng,
        runs=args.subsample_runs,
        frac=args.subsample_frac,
        null_model=args.null_model,
    )

    criteria = {
        "dominant_persistence_significant": bool(p_value_dom < args.alpha),
        "significant_count_significant": bool(p_value_count < args.alpha),
        "linearity_dims_enriched": bool(p_top_vs_rand < args.alpha),
        "subsample_more_stable_than_null": bool(
            stability["obs_dominant_cv"] < stability["null_dominant_cv"]
        ),
    }

    score = int(sum(criteria.values()))
    if score >= 3:
        verdict = "支持“审计环为结构化信号而非随机噪声”。"
    elif score == 2:
        verdict = "部分支持结构化信号，建议增加 n-null 与样本量复核。"
    else:
        verdict = "证据不足，暂无法排除随机噪声解释。"

    result = {
        "config": {
            "features": args.features,
            "key": args.key,
            "k": args.k,
            "n_null": args.n_null,
            "null_model": args.null_model,
            "null_quantile": args.null_quantile,
            "alpha": args.alpha,
            "seed": args.seed,
            "max_points": args.max_points,
            "l2_normalize": bool(args.l2_normalize),
            "dim_slice": args.dim_slice,
            "dim_random_runs": args.dim_random_runs,
            "max_freq": args.max_freq,
            "subsample_runs": args.subsample_runs,
            "subsample_frac": args.subsample_frac,
        },
        "data": {
            "n_points_raw": int(n0),
            "n_dims_raw": int(d0),
            "n_points_used": int(n),
            "n_dims_used": int(d),
        },
        "observed": {
            "euclidean": eu,
            "geodesic": geo,
        },
        "null_distribution": {
            "persistence_threshold_from_null": None if not np.isfinite(threshold) else float(threshold),
            "dominant_mean": float(np.mean(null_dom_arr)),
            "dominant_std": float(np.std(null_dom_arr)),
            "dominant_95pct": float(np.quantile(null_dom_arr, 0.95)),
            "significant_count_mean": float(np.mean(null_sig_count)),
            "significant_count_95pct": float(np.quantile(null_sig_count, 0.95)),
        },
        "hypothesis_test": {
            "obs_dominant": obs_dom,
            "obs_significant_count": int(obs_sig_count),
            "p_value_dominant": p_value_dom,
            "z_score_dominant": z_score_dom,
            "p_value_significant_count": p_value_count,
        },
        "dimension_consistency": dim_consistency,
        "subsample_stability": stability,
        "criteria": criteria,
        "score": score,
        "verdict": verdict,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.plot:
        maybe_plot(
            path=args.plot,
            obs_dom=obs_dom,
            null_dom=null_dom_arr,
            obs_count=obs_sig_count,
            null_count=null_sig_count,
        )

    print("=" * 72)
    print("Loop Validation (Noise vs Structured Signal)")
    print("=" * 72)
    print(f"Data used: N={n}, D={d}")
    print(f"Observed geodesic dominant persistence: {obs_dom:.6f}")
    print(f"Observed significant H1 count: {obs_sig_count}")
    print(f"Null threshold (q={args.null_quantile}): {threshold if np.isfinite(threshold) else 'inf'}")
    print(f"p-value dominant: {p_value_dom:.6g}")
    print(f"p-value significant_count: {p_value_count:.6g}")
    if dim_consistency.get("available", False):
        print(f"p-value top-linearity vs random dims: {dim_consistency['p_top_vs_random']:.6g}")
    print(f"Stability CV(obs/null): {stability['obs_dominant_cv']:.4f} / {stability['null_dominant_cv']:.4f}")
    print(f"Score: {score}/4")
    print(f"Verdict: {verdict}")
    print(f"JSON saved to: {args.out}")
    if args.plot:
        print(f"Plot saved to: {args.plot}")
    print("=" * 72)


if __name__ == "__main__":
    main()
