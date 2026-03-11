"""
保存位置: Anla/experiments/nlp/visualize_v8.py

Byte-Level MLM v8 训练结果可视化脚本
=========================================================================

功能:
    1. 读取 complex/training_log.json 和 real/training_log.json
    2. 生成 6 类对比图表 (保存为 PNG):
       - Fig 1: Loss 曲线对比 (含平滑)
       - Fig 2: 训练准确率对比
       - Fig 3: 测试准确率对比
       - Fig 4: 测试 Perplexity 对比 (对数坐标)
       - Fig 5: 学习率调度
       - Fig 6: 综合仪表板 (2×3 子图 + 摘要表)
    3. 可选: 加载模型权重，分析 DCU 内部参数分布
       - 本振 c 的模长 & 相位分布
       - 门控阈值 b 的分布
       - Embedding 模长分布 (观察自发对称性破缺)

数据来源:
    byte_mlm_v8.py 训练时保存的 training_log.json，结构为:
    {
      "model": "complex" | "real",
      "best_train_acc": float,
      "final_test_acc": float,
      "final_test_ppl": float,
      "total_params": int,
      "total_real_params": int,
      "training_time_sec": float,
      "history": {
        "epochs": [int, ...],
        "loss": [float, ...],
        "train_acc": [float, ...],
        "test_acc": [float, ...],   // -1.0 表示该轮未评估
        "test_ppl": [float, ...],   // -1.0 表示该轮未评估
        "lr": [float, ...]
      },
      "config": { ... }
    }

用法:
    # 基本用法: 指定训练输出目录
    python -m Anla.experiments.nlp.visualize_v8 --log-dir Anla/Logs/nlp_byte_mlm_v8

    # 只可视化复数模型 (无对比)
    python -m Anla.experiments.nlp.visualize_v8 --log-dir Anla/Logs/nlp_byte_mlm_v8 --model complex

    # 附加 DCU 参数分析 (需要 best.pth 权重文件)
    python -m Anla.experiments.nlp.visualize_v8 --log-dir Anla/Logs/nlp_byte_mlm_v8 --analyze-weights

    # 指定输出目录
    python -m Anla.experiments.nlp.visualize_v8 --log-dir Anla/Logs/nlp_byte_mlm_v8 --output-dir ./figures

    # 调整平滑窗口
    python -m Anla.experiments.nlp.visualize_v8 --log-dir Anla/Logs/nlp_byte_mlm_v8 --smooth-window 20
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ---- Matplotlib 后端配置 (必须在 import pyplot 之前) ----
import matplotlib
matplotlib.use('Agg')  # 无头模式，适配服务器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ---- 全局样式配置 ----
# 使用学术论文风格: 白底、细网格、serif 字体
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.pad_inches': 0.15,
})

# ---- 配色方案 ----
# 复数模型: 蓝-紫色系 (呼应量子/干涉主题)
# 实数基线: 橙-红色系 (暖色调区分)
COLOR_COMPLEX = '#4A90D9'       # 主色: 钴蓝
COLOR_COMPLEX_FILL = '#4A90D9'  # 填充: 同色低透明度
COLOR_COMPLEX_SMOOTH = '#1B3A6B'  # 平滑线: 深蓝
COLOR_REAL = '#E8712B'          # 主色: 橙
COLOR_REAL_FILL = '#E8712B'     # 填充: 同色低透明度
COLOR_REAL_SMOOTH = '#8B3A0F'   # 平滑线: 深橙
COLOR_LR = '#2ECC71'            # 学习率: 绿色
COLOR_HIGHLIGHT = '#E74C3C'     # 高亮标注: 红色

# 模型显示名称
LABEL_COMPLEX = 'HoloDCU (ℂ)'
LABEL_REAL = 'Real Baseline (ℝ)'


# =====================================================================
#  数据加载与预处理
# =====================================================================

def load_training_log(log_path: str) -> Optional[Dict[str, Any]]:
    """
    加载单个模型的训练日志。

    Args:
        log_path: training_log.json 文件路径

    Returns:
        训练日志字典，若文件不存在则返回 None
    """
    if not os.path.exists(log_path):
        print(f"  [警告] 日志文件不存在: {log_path}")
        return None

    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  [加载] {log_path}")
    print(f"         模型: {data.get('model', 'unknown')}, "
          f"Epochs: {len(data['history']['epochs'])}, "
          f"最终 Test Acc: {data.get('final_test_acc', -1):.2%}")
    return data


def exponential_moving_average(values: List[float],
                                alpha: float = 0.1) -> np.ndarray:
    """
    指数移动平均 (EMA) 平滑。

    比简单滑动窗口更好的特性:
    - 无滞后偏移 (简单滑窗有 window/2 的滞后)
    - 对最近的数据点赋予更大权重
    - 输出长度与输入相同

    Args:
        values: 原始数据序列
        alpha: 平滑因子 (0~1), 越小越平滑

    Returns:
        平滑后的 numpy 数组
    """
    arr = np.array(values, dtype=np.float64)
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def filter_valid_points(epochs: List[int], values: List[float],
                        invalid_val: float = -1.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    过滤无效数据点 (test_acc 和 test_ppl 中 -1.0 表示该轮未评估)。

    Args:
        epochs: epoch 列表
        values: 对应的指标值列表
        invalid_val: 无效标记值

    Returns:
        (有效 epochs, 有效 values) 的 numpy 数组元组
    """
    ep = np.array(epochs)
    val = np.array(values)
    mask = val > invalid_val
    return ep[mask], val[mask]


# =====================================================================
#  单图绘制函数
# =====================================================================

def plot_loss_curves(logs: Dict[str, Dict], smooth_alpha: float,
                     output_path: str):
    """
    Fig 1: Loss 曲线对比。

    显示原始 loss (半透明散点/细线) + EMA 平滑线 (粗线)。
    训练 loss 是每个记录点的单 batch loss，波动较大，
    平滑线帮助观察整体趋势。

    Args:
        logs: {'complex': log_dict, 'real': log_dict} 字典
        smooth_alpha: EMA 平滑因子
        output_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, log in logs.items():
        if log is None:
            continue
        epochs = log['history']['epochs']
        losses = log['history']['loss']

        # 配色与标签
        if name == 'complex':
            color, color_smooth, label = (
                COLOR_COMPLEX, COLOR_COMPLEX_SMOOTH, LABEL_COMPLEX)
        else:
            color, color_smooth, label = (
                COLOR_REAL, COLOR_REAL_SMOOTH, LABEL_REAL)

        # 原始 loss: 细线 + 低透明度 (展示真实波动)
        ax.plot(epochs, losses, color=color, alpha=0.25, linewidth=0.6)

        # EMA 平滑线: 粗线 (展示趋势)
        smoothed = exponential_moving_average(losses, alpha=smooth_alpha)
        ax.plot(epochs, smoothed, color=color_smooth, linewidth=2.0,
                label=f'{label} (EMA α={smooth_alpha})')

        # 标注最终 loss 值
        ax.annotate(
            f'{smoothed[-1]:.3f}',
            xy=(epochs[-1], smoothed[-1]),
            xytext=(10, 0), textcoords='offset points',
            fontsize=9, color=color_smooth,
            arrowprops=dict(arrowstyle='->', color=color_smooth, lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss — HoloDCU (ℂ) vs Real Baseline (ℝ)')
    ax.legend(loc='upper right', framealpha=0.9)

    # Y 轴下限设为 0 (loss 非负)
    ax.set_ylim(bottom=0)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_path}")


def plot_train_accuracy(logs: Dict[str, Dict], smooth_alpha: float,
                        output_path: str):
    """
    Fig 2: 训练准确率对比。

    训练准确率是每个记录点的单 batch argmax accuracy。
    与 loss 类似，使用 EMA 平滑展示趋势。

    Args:
        logs: 日志字典
        smooth_alpha: EMA 平滑因子
        output_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, log in logs.items():
        if log is None:
            continue
        epochs = log['history']['epochs']
        accs = log['history']['train_acc']

        if name == 'complex':
            color, color_smooth, label = (
                COLOR_COMPLEX, COLOR_COMPLEX_SMOOTH, LABEL_COMPLEX)
        else:
            color, color_smooth, label = (
                COLOR_REAL, COLOR_REAL_SMOOTH, LABEL_REAL)

        # 原始准确率: 低透明度
        ax.plot(epochs, [a * 100 for a in accs],
                color=color, alpha=0.25, linewidth=0.6)

        # EMA 平滑
        smoothed = exponential_moving_average(
            [a * 100 for a in accs], alpha=smooth_alpha)
        ax.plot(epochs, smoothed, color=color_smooth, linewidth=2.0,
                label=f'{label} (EMA)')

        # 标注最终值
        ax.annotate(
            f'{smoothed[-1]:.1f}%',
            xy=(epochs[-1], smoothed[-1]),
            xytext=(10, 0), textcoords='offset points',
            fontsize=9, color=color_smooth,
            arrowprops=dict(arrowstyle='->', color=color_smooth, lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Accuracy (%)')
    ax.set_title('Training Accuracy — HoloDCU (ℂ) vs Real Baseline (ℝ)')
    ax.legend(loc='lower right', framealpha=0.9)

    # Y 轴: 0% ~ 100%
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_path}")


def plot_test_accuracy(logs: Dict[str, Dict], output_path: str):
    """
    Fig 3: 测试准确率对比。

    测试准确率只在部分 epoch 评估 (log_interval * 5 的间隔)，
    数据点较少，不做 EMA 平滑，用折线 + 标记点显示。

    Args:
        logs: 日志字典
        output_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, log in logs.items():
        if log is None:
            continue

        # 过滤 test_acc == -1 的无效点
        ep_valid, acc_valid = filter_valid_points(
            log['history']['epochs'], log['history']['test_acc'])

        if len(ep_valid) == 0:
            continue

        if name == 'complex':
            color, label = COLOR_COMPLEX, LABEL_COMPLEX
            marker = 'o'
        else:
            color, label = COLOR_REAL, LABEL_REAL
            marker = 's'

        ax.plot(ep_valid, acc_valid * 100,
                color=color, linewidth=1.8, marker=marker, markersize=5,
                label=label)

        # 标注最终值
        ax.annotate(
            f'{acc_valid[-1] * 100:.1f}%',
            xy=(ep_valid[-1], acc_valid[-1] * 100),
            xytext=(10, 5), textcoords='offset points',
            fontsize=9, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy — HoloDCU (ℂ) vs Real Baseline (ℝ)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_path}")


def plot_test_perplexity(logs: Dict[str, Dict], output_path: str):
    """
    Fig 4: 测试 Perplexity 对比 (对数坐标)。

    Perplexity = exp(CE Loss)，值域 [1, +∞)。
    使用对数 Y 轴更清晰地展示下降趋势。

    Args:
        logs: 日志字典
        output_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, log in logs.items():
        if log is None:
            continue

        ep_valid, ppl_valid = filter_valid_points(
            log['history']['epochs'], log['history']['test_ppl'])

        if len(ep_valid) == 0:
            continue

        if name == 'complex':
            color, label = COLOR_COMPLEX, LABEL_COMPLEX
            marker = 'o'
        else:
            color, label = COLOR_REAL, LABEL_REAL
            marker = 's'

        ax.plot(ep_valid, ppl_valid,
                color=color, linewidth=1.8, marker=marker, markersize=5,
                label=label)

        # 标注最终值
        ax.annotate(
            f'PPL={ppl_valid[-1]:.1f}',
            xy=(ep_valid[-1], ppl_valid[-1]),
            xytext=(10, 5), textcoords='offset points',
            fontsize=9, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Perplexity (log scale)')
    ax.set_title('Test Perplexity — HoloDCU (ℂ) vs Real Baseline (ℝ)')
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.9)

    # 参考线: PPL = 256 (随机猜测的 perplexity)
    ax.axhline(y=256, color='gray', linestyle=':', alpha=0.6, linewidth=1.0)
    ax.text(ax.get_xlim()[0] + 50, 256 * 1.1, 'Random (PPL=256)',
            fontsize=9, color='gray', alpha=0.7)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_path}")


def plot_learning_rate(logs: Dict[str, Dict], output_path: str):
    """
    Fig 5: 学习率调度。

    两个模型使用相同的 LR 调度 (线性 warmup)，
    只需绘制一条曲线。若不同则绘制两条。

    Args:
        logs: 日志字典
        output_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for name, log in logs.items():
        if log is None:
            continue
        epochs = log['history']['epochs']
        lrs = log['history']['lr']

        if name == 'complex':
            color, label = COLOR_COMPLEX, LABEL_COMPLEX
        else:
            color, label = COLOR_REAL, LABEL_REAL

        ax.plot(epochs, lrs, color=color, linewidth=1.5, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_path}")


def plot_dashboard(logs: Dict[str, Dict], smooth_alpha: float,
                   output_path: str):
    """
    Fig 6: 综合仪表板 (2 行 × 3 列子图 + 摘要信息)。

    将所有关键指标汇总在一张大图中:
        [0,0] Loss 曲线     [0,1] 训练准确率    [0,2] 学习率
        [1,0] 测试准确率    [1,1] 测试 PPL      [1,2] 摘要表格

    Args:
        logs: 日志字典
        smooth_alpha: EMA 平滑因子
        output_path: 图片保存路径
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- [0,0] Loss 曲线 ----
    ax_loss = fig.add_subplot(gs[0, 0])
    for name, log in logs.items():
        if log is None:
            continue
        ep = log['history']['epochs']
        loss = log['history']['loss']
        if name == 'complex':
            c, cs, lb = COLOR_COMPLEX, COLOR_COMPLEX_SMOOTH, LABEL_COMPLEX
        else:
            c, cs, lb = COLOR_REAL, COLOR_REAL_SMOOTH, LABEL_REAL
        ax_loss.plot(ep, loss, color=c, alpha=0.2, linewidth=0.5)
        sm = exponential_moving_average(loss, alpha=smooth_alpha)
        ax_loss.plot(ep, sm, color=cs, linewidth=1.5, label=lb)
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylim(bottom=0)
    ax_loss.legend(fontsize=8)

    # ---- [0,1] 训练准确率 ----
    ax_train = fig.add_subplot(gs[0, 1])
    for name, log in logs.items():
        if log is None:
            continue
        ep = log['history']['epochs']
        acc = [a * 100 for a in log['history']['train_acc']]
        if name == 'complex':
            c, cs, lb = COLOR_COMPLEX, COLOR_COMPLEX_SMOOTH, LABEL_COMPLEX
        else:
            c, cs, lb = COLOR_REAL, COLOR_REAL_SMOOTH, LABEL_REAL
        ax_train.plot(ep, acc, color=c, alpha=0.2, linewidth=0.5)
        sm = exponential_moving_average(acc, alpha=smooth_alpha)
        ax_train.plot(ep, sm, color=cs, linewidth=1.5, label=lb)
    ax_train.set_title('Train Accuracy')
    ax_train.set_xlabel('Epoch')
    ax_train.set_ylim(0, 105)
    ax_train.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax_train.legend(fontsize=8)

    # ---- [0,2] 学习率 ----
    ax_lr = fig.add_subplot(gs[0, 2])
    for name, log in logs.items():
        if log is None:
            continue
        ep = log['history']['epochs']
        lr = log['history']['lr']
        if name == 'complex':
            c, lb = COLOR_COMPLEX, LABEL_COMPLEX
        else:
            c, lb = COLOR_REAL, LABEL_REAL
        ax_lr.plot(ep, lr, color=c, linewidth=1.2, label=lb)
    ax_lr.set_title('Learning Rate')
    ax_lr.set_xlabel('Epoch')
    ax_lr.legend(fontsize=8)

    # ---- [1,0] 测试准确率 ----
    ax_test = fig.add_subplot(gs[1, 0])
    for name, log in logs.items():
        if log is None:
            continue
        ep_v, acc_v = filter_valid_points(
            log['history']['epochs'], log['history']['test_acc'])
        if len(ep_v) == 0:
            continue
        if name == 'complex':
            c, lb, mk = COLOR_COMPLEX, LABEL_COMPLEX, 'o'
        else:
            c, lb, mk = COLOR_REAL, LABEL_REAL, 's'
        ax_test.plot(ep_v, acc_v * 100, color=c, linewidth=1.5,
                     marker=mk, markersize=4, label=lb)
    ax_test.set_title('Test Accuracy')
    ax_test.set_xlabel('Epoch')
    ax_test.set_ylim(0, 105)
    ax_test.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax_test.legend(fontsize=8)

    # ---- [1,1] 测试 Perplexity ----
    ax_ppl = fig.add_subplot(gs[1, 1])
    for name, log in logs.items():
        if log is None:
            continue
        ep_v, ppl_v = filter_valid_points(
            log['history']['epochs'], log['history']['test_ppl'])
        if len(ep_v) == 0:
            continue
        if name == 'complex':
            c, lb, mk = COLOR_COMPLEX, LABEL_COMPLEX, 'o'
        else:
            c, lb, mk = COLOR_REAL, LABEL_REAL, 's'
        ax_ppl.plot(ep_v, ppl_v, color=c, linewidth=1.5,
                    marker=mk, markersize=4, label=lb)
    ax_ppl.set_title('Test Perplexity')
    ax_ppl.set_xlabel('Epoch')
    ax_ppl.set_yscale('log')
    ax_ppl.axhline(y=256, color='gray', linestyle=':', alpha=0.5)
    ax_ppl.legend(fontsize=8)

    # ---- [1,2] 摘要表格 ----
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis('off')

    # 构建表格数据
    table_data = []
    col_labels = ['Metric', ]
    for name, log in logs.items():
        if log is None:
            continue
        col_labels.append(LABEL_COMPLEX if name == 'complex' else LABEL_REAL)

    # 准备行数据
    row_metrics = [
        ('Best Train Acc', 'best_train_acc', lambda v: f'{v:.2%}'),
        ('Final Test Acc', 'final_test_acc', lambda v: f'{v:.2%}'),
        ('Final Test PPL', 'final_test_ppl', lambda v: f'{v:.1f}'),
        ('Params (total)', 'total_params', lambda v: f'{v:,}'),
        ('Params (real equiv)', 'total_real_params', lambda v: f'{v:,}'),
        ('Training Time', 'training_time_sec', lambda v: f'{v:.0f}s'),
    ]

    for metric_label, key, fmt in row_metrics:
        row = [metric_label]
        for name in logs:
            log = logs[name]
            if log is None:
                row.append('N/A')
            elif key in log:
                row.append(fmt(log[key]))
            else:
                row.append('N/A')
        table_data.append(row)

    if len(table_data) > 0 and len(col_labels) > 1:
        table = ax_summary.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)

        # 表头样式
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(fontweight='bold')

        # 交替行背景色
        for i in range(len(table_data)):
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                if i % 2 == 0:
                    cell.set_facecolor('#F8F8F8')

    ax_summary.set_title('Summary', fontsize=13, fontweight='bold', pad=15)

    # 总标题
    fig.suptitle(
        'Byte-MLM v8 Training Dashboard — '
        'HolographicAttention + DCU-FFN',
        fontsize=15, fontweight='bold', y=0.98)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {output_path}")


# =====================================================================
#  DCU 参数分析 (可选，需要模型权重)
# =====================================================================

def analyze_dcu_weights(weight_path: str, output_dir: str,
                        d_model: int = 64, num_blocks: int = 3,
                        ff_mult: int = 4):
    """
    分析 DCU-FFN 的内部参数分布。

    从 best.pth 加载训练后的权重，绘制:
    1. 本振 c 的模长分布 (每层一个子图)
       → 观察是否学到了非均匀增益 (自发打破初始化的单位模长)
    2. 本振 c 的相位分布 (极坐标玫瑰图)
       → 观察是否形成了特定的方向偏好 (相位聚类)
    3. 门控阈值 b 的分布 (每层一个子图)
       → 观察有多少通道被「默认关闭」(b >> 0) 或「默认开启」(b << 0)
    4. Embedding 模长分布 (排除 [MASK] token)
       → 观察自发对称性破缺: 高频 token 是否学到更大的模长

    Args:
        weight_path: best.pth 文件路径
        output_dir: 图片保存目录
        d_model: 复数维度 D
        num_blocks: Transformer 层数 L
        ff_mult: FFN 扩展倍数
    """
    import torch

    if not os.path.exists(weight_path):
        print(f"  [跳过] 权重文件不存在: {weight_path}")
        return

    print(f"  [加载] 权重: {weight_path}")
    state = torch.load(weight_path, map_location='cpu', weights_only=True)

    ff_dim = d_model * ff_mult

    # ================================================================
    # Fig A: 本振 c 的模长分布 (每层一个子图)
    # ================================================================
    fig_c_mag, axes_c_mag = plt.subplots(
        1, num_blocks, figsize=(5 * num_blocks, 4), squeeze=False)

    for layer_idx in range(num_blocks):
        key = f'blocks.{layer_idx}.ffn.c'
        if key not in state:
            print(f"  [警告] 未找到参数: {key}")
            continue

        c = state[key]  # ℂ^M
        c_mag = c.abs().numpy()

        ax = axes_c_mag[0, layer_idx]
        ax.hist(c_mag, bins=40, color=COLOR_COMPLEX, alpha=0.7,
                edgecolor='white', linewidth=0.5)
        ax.axvline(x=1.0, color=COLOR_HIGHLIGHT, linestyle='--',
                   linewidth=1.0, label='Init |c|=1')
        ax.set_title(f'Layer {layer_idx}: |c| Distribution')
        ax.set_xlabel('|c_m| (LO Magnitude)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

        # 统计信息
        ax.text(0.95, 0.95,
                f'mean={c_mag.mean():.3f}\n'
                f'std={c_mag.std():.3f}\n'
                f'min={c_mag.min():.3f}\n'
                f'max={c_mag.max():.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_c_mag.suptitle('DCU Local Oscillator |c| Distribution (per layer)', fontweight='bold')
    path_c_mag = os.path.join(output_dir, 'dcu_c_magnitude.png')
    fig_c_mag.savefig(path_c_mag, bbox_inches='tight')
    plt.close(fig_c_mag)
    print(f"  [保存] {path_c_mag}")

    # ================================================================
    # Fig B: 本振 c 的相位分布 (极坐标玫瑰图)
    # ================================================================
    fig_c_phase, axes_c_phase = plt.subplots(
        1, num_blocks, figsize=(5 * num_blocks, 5),
        subplot_kw={'projection': 'polar'}, squeeze=False)

    for layer_idx in range(num_blocks):
        key = f'blocks.{layer_idx}.ffn.c'
        if key not in state:
            continue

        c = state[key]
        phases = torch.angle(c).numpy()  # [-π, π]

        ax = axes_c_phase[0, layer_idx]
        # 玫瑰图: 将相位分成 36 个 bin (每 10°)
        n_bins = 36
        bins_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        counts, _ = np.histogram(phases, bins=bins_edges)
        # 每个 bin 的中心角度
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        width = 2 * np.pi / n_bins

        ax.bar(bin_centers, counts, width=width,
               color=COLOR_COMPLEX, alpha=0.7, edgecolor='white')
        ax.set_title(f'Layer {layer_idx}: arg(c) Rose Plot',
                     pad=15, fontsize=11)
        ax.set_yticklabels([])  # hide radial ticks

    fig_c_phase.suptitle('DCU Local Oscillator Phase arg(c) Distribution (per layer)',
                         fontweight='bold', y=1.02)
    path_c_phase = os.path.join(output_dir, 'dcu_c_phase.png')
    fig_c_phase.savefig(path_c_phase, bbox_inches='tight')
    plt.close(fig_c_phase)
    print(f"  [保存] {path_c_phase}")

    # ================================================================
    # Fig C: 门控阈值 b 的分布
    # ================================================================
    fig_b, axes_b = plt.subplots(
        1, num_blocks, figsize=(5 * num_blocks, 4), squeeze=False)

    for layer_idx in range(num_blocks):
        key = f'blocks.{layer_idx}.ffn.b'
        if key not in state:
            continue

        b = state[key].numpy()

        ax = axes_b[0, layer_idx]
        ax.hist(b, bins=40, color='#9B59B6', alpha=0.7,
                edgecolor='white', linewidth=0.5)
        ax.axvline(x=0, color=COLOR_HIGHLIGHT, linestyle='--',
                   linewidth=1.0, label='Init b=0')
        ax.set_title(f'Layer {layer_idx}: b Distribution')
        ax.set_xlabel('b_m (Gate Threshold)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

        # 统计: 有多少通道倾向于「关闭」(b > 0.5)
        n_closed = np.sum(b > 0.5)
        n_open = np.sum(b < -0.5)
        ax.text(0.95, 0.95,
                f'mean={b.mean():.3f}\n'
                f'std={b.std():.3f}\n'
                f'b>0.5: {n_closed}/{len(b)}\n'
                f'b<-0.5: {n_open}/{len(b)}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_b.suptitle('DCU Gate Threshold b Distribution (per layer)', fontweight='bold')
    path_b = os.path.join(output_dir, 'dcu_b_threshold.png')
    fig_b.savefig(path_b, bbox_inches='tight')
    plt.close(fig_b)
    print(f"  [保存] {path_b}")

    # ================================================================
    # Fig D: Embedding 模长分布 — 自发对称性破缺观测
    # ================================================================
    emb_key = 'embedding.weight'
    if emb_key in state:
        E = state[emb_key]  # ℂ^{(V+1) × D}
        # 排除 [MASK] token (最后一行)
        E_vocab = E[:-1]  # ℂ^{V × D}
        emb_norms = E_vocab.abs().pow(2).sum(dim=-1).sqrt().numpy()  # ℝ^V

        fig_emb, (ax_hist, ax_scatter) = plt.subplots(
            1, 2, figsize=(14, 5))

        # 左: 模长直方图
        ax_hist.hist(emb_norms, bins=50, color=COLOR_COMPLEX, alpha=0.7,
                     edgecolor='white', linewidth=0.5)
        # 理论初始化 RMS: √(D × 1/D) = 1.0
        ax_hist.axvline(x=1.0, color=COLOR_HIGHLIGHT, linestyle='--',
                        linewidth=1.0, label='Init RMS approx 1.0')
        ax_hist.set_xlabel('‖E[v]‖ (Embedding Magnitude)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Embedding Magnitude Distribution')
        ax_hist.legend(fontsize=9)
        ax_hist.text(0.95, 0.95,
                     f'mean={emb_norms.mean():.3f}\n'
                     f'std={emb_norms.std():.3f}\n'
                     f'max/min={emb_norms.max()/emb_norms.min():.2f}×',
                     transform=ax_hist.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 右: 按 token id 排列的模长散点图
        # 标注模长最大 / 最小的几个 token
        ax_scatter.scatter(range(len(emb_norms)), emb_norms,
                           s=8, color=COLOR_COMPLEX, alpha=0.6)
        ax_scatter.set_xlabel('Token ID (byte value)')
        ax_scatter.set_ylabel('‖E[v]‖')
        ax_scatter.set_title('Embedding Magnitude vs Token ID')

        # 标注 Top-5 最大模长的 token
        top5_idx = np.argsort(emb_norms)[-5:]
        for idx in top5_idx:
            char_repr = repr(chr(idx)) if 32 <= idx < 127 else f'0x{idx:02X}'
            ax_scatter.annotate(
                char_repr,
                xy=(idx, emb_norms[idx]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, color=COLOR_HIGHLIGHT,
                arrowprops=dict(arrowstyle='->', color=COLOR_HIGHLIGHT,
                                lw=0.6))

        # 标注 Bottom-5 最小模长的 token
        bot5_idx = np.argsort(emb_norms)[:5]
        for idx in bot5_idx:
            char_repr = repr(chr(idx)) if 32 <= idx < 127 else f'0x{idx:02X}'
            ax_scatter.annotate(
                char_repr,
                xy=(idx, emb_norms[idx]),
                xytext=(5, -10), textcoords='offset points',
                fontsize=7, color='#7F8C8D',
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=0.6))

        fig_emb.suptitle(
            'Embedding Magnitude Analysis — Spontaneous Symmetry Breaking',
            fontweight='bold')
        path_emb = os.path.join(output_dir, 'embedding_magnitude.png')
        fig_emb.savefig(path_emb, bbox_inches='tight')
        plt.close(fig_emb)
        print(f"  [保存] {path_emb}")
    else:
        print(f"  [警告] 未找到 embedding 权重: {emb_key}")

    # ================================================================
    # Fig E: 综合 DCU 仪表板 (c 散点图: 模长 vs 相位)
    # ================================================================
    fig_dcu_all, axes_dcu = plt.subplots(
        1, num_blocks, figsize=(5 * num_blocks, 5),
        subplot_kw={'projection': 'polar'}, squeeze=False)

    for layer_idx in range(num_blocks):
        key = f'blocks.{layer_idx}.ffn.c'
        if key not in state:
            continue

        c = state[key]
        mags = c.abs().numpy()
        phases = torch.angle(c).numpy()

        ax = axes_dcu[0, layer_idx]
        # 极坐标散点图: 每个点是一个本振
        # 颜色编码模长 (偏离初始化 1.0 的程度)
        scatter = ax.scatter(phases, mags, c=mags, cmap='coolwarm',
                             s=12, alpha=0.7, vmin=0.5, vmax=1.5)
        ax.set_title(f'Layer {layer_idx}: c Distribution (r=|c|, theta=arg(c))',
                     pad=15, fontsize=10)

        # 画初始化参考圆 (|c|=1)
        theta_ref = np.linspace(0, 2 * np.pi, 100)
        ax.plot(theta_ref, np.ones_like(theta_ref), '--',
                color='gray', alpha=0.4, linewidth=0.8)

    # 添加颜色条
    if num_blocks > 0:
        fig_dcu_all.colorbar(
            scatter, ax=axes_dcu.ravel().tolist(),
            label='|c| (LO Magnitude)', shrink=0.8)

    fig_dcu_all.suptitle(
        'DCU Local Oscillator c — Polar Distribution (per layer)',
        fontweight='bold', y=1.02)
    path_dcu_all = os.path.join(output_dir, 'dcu_c_polar.png')
    fig_dcu_all.savefig(path_dcu_all, bbox_inches='tight')
    plt.close(fig_dcu_all)
    print(f"  [保存] {path_dcu_all}")


# =====================================================================
#  主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Byte-MLM v8 训练结果可视化")
    parser.add_argument(
        '--log-dir', type=str, required=True,
        help="训练输出根目录 (包含 complex/ 和 real/ 子目录)")
    parser.add_argument(
        '--model', type=str, default='both',
        choices=['complex', 'real', 'both'],
        help="要可视化的模型 (默认: both)")
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help="图片保存目录 (默认: {log-dir}/figures)")
    parser.add_argument(
        '--smooth-alpha', type=float, default=0.05,
        help="EMA 平滑因子 (0~1, 越小越平滑, 默认: 0.05)")
    parser.add_argument(
        '--analyze-weights', action='store_true',
        help="是否分析 DCU 内部权重分布 (需要 best.pth)")

    args = parser.parse_args()

    # ---- 输出目录 ----
    output_dir = args.output_dir or os.path.join(args.log_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # ---- 加载日志 ----
    logs: Dict[str, Optional[Dict]] = {}

    if args.model in ('complex', 'both'):
        logs['complex'] = load_training_log(
            os.path.join(args.log_dir, 'complex', 'training_log.json'))

    if args.model in ('real', 'both'):
        logs['real'] = load_training_log(
            os.path.join(args.log_dir, 'real', 'training_log.json'))

    # 过滤掉加载失败的
    active_logs = {k: v for k, v in logs.items() if v is not None}

    if not active_logs:
        print("[错误] 没有找到任何有效的训练日志。请检查 --log-dir 路径。")
        sys.exit(1)

    print(f"\n加载了 {len(active_logs)} 个模型的日志: "
          f"{list(active_logs.keys())}")

    # ---- 生成图表 ----
    print("\n" + "=" * 60)
    print("  生成图表")
    print("=" * 60)

    # Fig 1: Loss 曲线
    plot_loss_curves(
        active_logs, args.smooth_alpha,
        os.path.join(output_dir, '01_loss.png'))

    # Fig 2: 训练准确率
    plot_train_accuracy(
        active_logs, args.smooth_alpha,
        os.path.join(output_dir, '02_train_accuracy.png'))

    # Fig 3: 测试准确率
    plot_test_accuracy(
        active_logs,
        os.path.join(output_dir, '03_test_accuracy.png'))

    # Fig 4: 测试 Perplexity
    plot_test_perplexity(
        active_logs,
        os.path.join(output_dir, '04_test_perplexity.png'))

    # Fig 5: 学习率
    plot_learning_rate(
        active_logs,
        os.path.join(output_dir, '05_learning_rate.png'))

    # Fig 6: 综合仪表板
    plot_dashboard(
        active_logs, args.smooth_alpha,
        os.path.join(output_dir, '06_dashboard.png'))

    # ---- 可选: DCU 权重分析 ----
    if args.analyze_weights:
        print("\n" + "=" * 60)
        print("  DCU 权重分析")
        print("=" * 60)

        weight_path = os.path.join(args.log_dir, 'complex', 'best.pth')

        # 从日志中读取模型超参数
        complex_log = logs.get('complex')
        if complex_log and 'config' in complex_log:
            d_model = complex_log['config'].get('d_model_complex', 64)
            num_blocks = complex_log['config'].get('num_blocks', 3)
            ff_mult = complex_log['config'].get('ff_mult', 4)
        else:
            d_model, num_blocks, ff_mult = 64, 3, 4

        analyze_dcu_weights(
            weight_path, output_dir,
            d_model=d_model, num_blocks=num_blocks, ff_mult=ff_mult)

    # ---- 完成 ----
    n_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"\n{'=' * 60}")
    print(f"  完成! 共生成 {n_files} 张图表")
    print(f"  保存位置: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
