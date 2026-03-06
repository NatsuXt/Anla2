"""
功能:
    从训练好的 checkpoint 中提取原始复数 embedding 权重,
    直接保存为 validate_tda_loops.py 可读取的 .npz 格式。

    这绕过了 capture_ring_features_from_audit.py 的 MDS 重建步骤,
    避免了信息有损变换对拓扑分析的干扰。

输出:
    .npz 文件, 包含两个键:
      - z:      (vocab_size, d_model) complex128 — 原始复数 embedding
      - x_real: (vocab_size, 2*d_model) float64  — 实虚拼接形式 (冗余, 供兼容)

用法:
    python -m Anla.extract_ring_embedding \
        --checkpoint checkpoints/best_ring_inpainter.pth \
        --vocab_size 64 \
        --output Logs/ring_masking_vis/ring_features.npz
"""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("此脚本需要 PyTorch。请运行: pip install torch") from exc


def find_embedding_key(state_dict: dict) -> str:
    """
    在 state_dict 中查找 embedding 权重的键名。

    搜索策略:
      1. 精确匹配常见键名
      2. 后缀匹配 (适配不同的模型封装层级)
      3. 全部失败则报错并列出可用键
    """
    # 精确匹配候选
    exact_candidates = [
        "embedding.weight",
        "embed.weight",
        "model.embedding.weight",
        "model.embed.weight",
    ]
    for key in exact_candidates:
        if key in state_dict:
            return key

    # 后缀匹配
    for key in state_dict.keys():
        if key.endswith("embedding.weight") or key.endswith("embed.weight"):
            return key

    # 未找到
    available = list(state_dict.keys())[:20]
    raise KeyError(
        f"无法在 checkpoint 中找到 embedding 权重。\n"
        f"已检查的键名: {exact_candidates}\n"
        f"checkpoint 中可用的键 (前 20 个): {available}"
    )


def extract_embedding(
    checkpoint_path: str,
    vocab_size: int = 64,
) -> np.ndarray:
    """
    从 checkpoint 加载 embedding 权重并返回原始复数矩阵。

    Parameters
    ----------
    checkpoint_path : str
        checkpoint 文件路径 (.pth 或 .pt)
    vocab_size : int
        词汇表大小 (不含 MASK token)。
        embedding 层实际大小为 vocab_size + 1,
        这里只提取前 vocab_size 行。

    Returns
    -------
    z : np.ndarray, shape (vocab_size, d_model), dtype complex128
        原始复数 embedding 权重
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 提取 state_dict (兼容直接保存 state_dict 或 包装在 dict 中的情况)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # 可能直接就是 state_dict
        state_dict = ckpt
    else:
        raise ValueError(
            f"Checkpoint 格式不支持: type={type(ckpt)}。"
            f"期望 dict 包含 'model_state_dict' 键。"
        )

    # 查找 embedding 键
    key = find_embedding_key(state_dict)
    weight = state_dict[key]

    # 转换为 numpy
    if hasattr(weight, "detach"):
        weight = weight.detach().cpu()
    weight_np = np.asarray(weight)

    # 验证复数类型
    if not np.iscomplexobj(weight_np):
        raise ValueError(
            f"Embedding 权重不是复数类型 (dtype={weight_np.dtype})。\n"
            f"此脚本仅适用于 Anla 的复数 embedding。"
        )

    # 验证行数
    if weight_np.shape[0] < vocab_size:
        raise ValueError(
            f"Embedding 行数 ({weight_np.shape[0]}) < vocab_size ({vocab_size})。\n"
            f"请检查 --vocab_size 参数。"
        )

    # 截取前 vocab_size 行 (排除 MASK token)
    z = weight_np[:vocab_size].astype(np.complex128)

    return z


def main():
    parser = argparse.ArgumentParser(
        description="从 checkpoint 提取原始复数 embedding, "
                    "保存为 validate_tda_loops.py 可直接读取的 .npz 格式。"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join("checkpoints", "best_ring_inpainter.pth"),
        help="checkpoint 文件路径 (默认: checkpoints/best_ring_inpainter.pth)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=64,
        help="词汇表大小, 不含 MASK token (默认: 64)"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join("Logs", "ring_masking_vis", "ring_features.npz"),
        help="输出 .npz 文件路径 (默认: Logs/ring_masking_vis/ring_features.npz)"
    )
    args = parser.parse_args()

    # ---- 提取 ----
    print("=" * 64)
    print("Extract Ring Embedding (原始复数权重)")
    print("=" * 64)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocab size: {args.vocab_size}")

    z = extract_embedding(args.checkpoint, args.vocab_size)

    n, d = z.shape
    print(f"Embedding shape: ({n}, {d})")
    print(f"Dtype: {z.dtype}")
    print(f"模长范围: [{np.abs(z).min():.6f}, {np.abs(z).max():.6f}]")
    print(f"模长均值: {np.abs(z).mean():.6f}")

    # ---- 保存 ----
    # 同时保存复数原始形式 (z) 和实虚拼接形式 (x_real)
    x_real = np.concatenate([z.real, z.imag], axis=1).astype(np.float64)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.output, z=z, x_real=x_real)

    print(f"\n已保存: {args.output}")
    print(f"  键 'z':      shape={z.shape}, dtype={z.dtype}")
    print(f"  键 'x_real': shape={x_real.shape}, dtype={x_real.dtype}")
    print()
    print("下一步: 运行零假设检验")
    print(f"  python -m Anla.validate_tda_loops \\")
    print(f"      --features {args.output} \\")
    print(f"      --key z \\")
    print(f"      --k 6 \\")
    print(f"      --n-null 300 \\")
    print(f"      --null-model phase_permute \\")
    print(f"      --out LOOP_VALIDATION.json \\")
    print(f"      --plot LOOP_VALIDATION.png")
    print("=" * 64)


if __name__ == "__main__":
    main()

