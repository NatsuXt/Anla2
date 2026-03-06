import argparse
import json
import os
import runpy
import sys
from typing import Any, Dict, List

import numpy as np
import ripser as ripser_module


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def is_distance_like(arr: np.ndarray) -> bool:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    if arr.size == 0:
        return False
    diag_small = float(np.mean(np.abs(np.diag(arr)))) < 1e-8
    sym = np.allclose(arr, arr.T, rtol=1e-4, atol=1e-6)
    nonneg = np.nanmin(arr) >= -1e-8
    return bool(diag_small and sym and nonneg)


def pair_real_to_complex(x_real: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(x_real):
        z = np.asarray(x_real, dtype=np.complex128)
        if z.ndim == 1:
            z = z[:, None]
        return z

    x_real = np.asarray(x_real, dtype=np.float64)
    if x_real.ndim == 1:
        x_real = x_real[:, None]
    if x_real.shape[1] % 2 == 1:
        pad = np.zeros((x_real.shape[0], 1), dtype=x_real.dtype)
        x_real = np.concatenate([x_real, pad], axis=1)
    return x_real[:, 0::2] + 1j * x_real[:, 1::2]


def classical_mds(D: np.ndarray, n_components: int = 32) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    D2 = D * D
    J = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * (J @ D2 @ J)

    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos = evals > 1e-10
    evals = evals[pos]
    evecs = evecs[:, pos]

    if evals.size == 0:
        return np.zeros((n, 1), dtype=np.float64)

    m = min(n_components, evals.size)
    return evecs[:, :m] * np.sqrt(evals[:m])[None, :]


def run_audit_and_capture(module_name: str, audit_args: List[str]) -> List[Dict[str, Any]]:
    captured: List[Dict[str, Any]] = []
    original_ripser = ripser_module.ripser

    def wrapped_ripser(X, *args, **kwargs):
        arr = to_numpy(X)
        distance_matrix = bool(kwargs.get("distance_matrix", False))
        if arr.ndim == 2:
            captured.append(
                {
                    "distance_matrix": distance_matrix,
                    "shape": [int(arr.shape[0]), int(arr.shape[1])],
                    "dtype": str(arr.dtype),
                    "array": arr.copy(),
                }
            )
        return original_ripser(X, *args, **kwargs)

    old_argv = sys.argv[:]
    ripser_module.ripser = wrapped_ripser
    try:
        sys.argv = [module_name] + audit_args
        runpy.run_module(module_name, run_name="__main__")
    finally:
        ripser_module.ripser = original_ripser
        sys.argv = old_argv

    return captured


def pick_feature_matrix(captured: List[Dict[str, Any]]) -> Dict[str, Any]:
    point_candidates = []
    dm_candidates = []

    for i, item in enumerate(captured):
        arr = item["array"]
        if item["distance_matrix"] or is_distance_like(arr):
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                dm_candidates.append((i, item))
        else:
            if arr.ndim == 2 and arr.shape[0] >= 8 and arr.shape[1] >= 1:
                point_candidates.append((i, item))

    if point_candidates:
        idx, best = max(point_candidates, key=lambda kv: kv[1]["shape"][0] * kv[1]["shape"][1])
        x_real = np.asarray(best["array"])
        z = pair_real_to_complex(x_real)
        return {
            "source": "ripser_input_point_cloud",
            "call_index": idx,
            "x_real": x_real,
            "z": z,
        }

    if dm_candidates:
        idx, best = max(dm_candidates, key=lambda kv: kv[1]["shape"][0])
        D = np.asarray(best["array"], dtype=np.float64)
        X = classical_mds(D, n_components=min(32, max(2, D.shape[0] - 1)))
        z = pair_real_to_complex(X)
        return {
            "source": "mds_from_distance_matrix",
            "call_index": idx,
            "x_real": X,
            "z": z,
        }

    raise RuntimeError("未捕获到可用的 ripser 输入矩阵。")


def main():
    parser = argparse.ArgumentParser(description="运行 topology_audit_ring 并导出 TDA 特征。")
    parser.add_argument("--module", type=str, default="Anla.topology_audit_ring")
    parser.add_argument("--out-features", type=str, required=True, help="输出 .npz（包含 key: z, x_real）")
    parser.add_argument("--out-meta", type=str, default="", help="输出元信息 JSON")
    parser.add_argument("audit_args", nargs=argparse.REMAINDER, help="传给审计模块的参数（建议用 -- 分隔）")
    args = parser.parse_args()

    audit_args = args.audit_args
    if audit_args and audit_args[0] == "--":
        audit_args = audit_args[1:]

    captured = run_audit_and_capture(args.module, audit_args)
    picked = pick_feature_matrix(captured)

    out_dir = os.path.dirname(args.out_features)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.out_features, z=picked["z"], x_real=picked["x_real"])

    meta = {
        "module": args.module,
        "audit_args": audit_args,
        "picked_source": picked["source"],
        "picked_call_index": int(picked["call_index"]),
        "picked_z_shape": [int(picked["z"].shape[0]), int(picked["z"].shape[1])],
        "captured_calls": [
            {
                "index": i,
                "distance_matrix": bool(item["distance_matrix"]),
                "shape": item["shape"],
                "dtype": item["dtype"],
            }
            for i, item in enumerate(captured)
        ],
        "n_captured_calls": len(captured),
    }

    if args.out_meta:
        meta_dir = os.path.dirname(args.out_meta)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)
        with open(args.out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Captured features for loop validation")
    print("=" * 72)
    print(f"Captured ripser calls: {len(captured)}")
    print(f"Selected source: {picked['source']}")
    print(f"Saved features: {args.out_features}")
    if args.out_meta:
        print(f"Saved meta: {args.out_meta}")
    print("=" * 72)


if __name__ == "__main__":
    main()
