"""
保存位置: Anla/experiments/capacity/run_config_B_train.py

Config B 训练启动器
=========================================================================

功能:
    以 Boltzmann-Elegant 损失训练 Config B (V=256, D=64)，
    并保存完整训练日志供后续可视化分析。

    本脚本是 capacity_pressure_test_v4.py 的轻量级包装器，
    不修改任何训练逻辑，仅指定运行参数。

用法:
    python -m Anla.experiments.capacity.run_config_B_train

输出目录:
    Logs/config_B_analysis/config_B_v256_d64/
        ├── training_log.json      ← 训练历史 (loss, acc, tau, p_tgt, neg%, ...)
        ├── best_checkpoint.p型权重
        └── ring_features.npz     ← 最终 embedding 矩阵 (complex128)
"""

import sys
import os

# -------------------------------------------------------------------------
# [Path Fix] 文件位置: Anla/experiments/capacity/run_config_B_train.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# =====================================================================
#  参数配置
# =====================================================================

# 输出根目录 (可视化脚本将从此目录读取数据)
OUTPUT_DIR = os.path.join(_ANLA_ROOT, "Logs", "config_B_analysis")

# 训练参数
CONFIGS = ["B"]              # 仅运行 Config B
LOSS_MODE = "boltzmann"      # 使用 v4 Boltzmann-Elegant 损失
EPOCHS = None                # None = 使用默认值 (10000 epochs)
PATIENCE = 2000              # Early stopping 容忍度
LOG_INTERVAL = 100           # 日志间隔 (比默认的 200 更密, 便于可视化曲线平滑)
DEVICE = "auto"              # 自动检测 GPU/CPU


# =====================================================================
#  构造命令行参数并调用训练入口
# =====================================================================
def main():
    """
    通过 sys.argv 注入参数，直接调用 capacity_pressure_test_v4.main()。
    这比 subprocess 更可靠，避免了路径和环境变量问题。
    """
    # 构造等效的命令行参数
    argv = [
        "capacity_pressure_test_v4",       # 程序名 (占位)
        "--configs"] + CONFIGS + [
        "--loss", LOSS_MODE,
        "--patience", str(PATIENCE),
        "--log-interval", str(LOG_INTERVAL),
        "--output-dir", OUTPUT_DIR,
        "--device", DEVICE,
    ]

    # 如果指定了 epochs 覆盖，追加参数
    if EPOCHS is not None:
        argv += ["--epochs", str(EPOCHS)]

    # 注入参数并调用
    sys.argv = argv

    print("=" * 72)
    print("  Config B 训练启动器")
    print("=" * 72)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  损失函数: {LOSS_MODE}")
    print(f"  日志间隔: {LOG_INTERVAL} epochs")
    print(f"  Early Stop 容忍度: {PATIENCE} epochs")
    print()

    from Anla.experiments.capacity.capacity_pressure_test_v4 import main as train_main
    train_main()

    print()
    print("=" * 72)
    print("  训练完成！")
    print(f"  结果保存在: {OUTPUT_DIR}/config_B_v256_d64/")
    print()
    print("  下一步: 运行可视化脚本")
    print("    python -m Anla.visualization.visualize_config_B")
    print("=" * 72)


if __name__ == "__main__":
    main()
