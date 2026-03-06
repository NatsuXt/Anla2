# Project Anla — 复数流形 Transformer

> 基于全纯复数计算的序列流形学习系统。  
> 以波干涉为注意力机制、以极坐标动力学为优化器、以 Boltzmann 统计力学为损失函数。

---

## 概述

Project Anla 构建了一个**全计算流程运行在复数域**的 Transformer 变体。不同于将复数作为后处理工具，Anla 将复数的代数结构——旋转、相位干涉、模长-相位解耦——作为架构设计的一等原语。

核心假设：当序列数据具有周期性或拓扑结构（如环、环面）时，复数流形嵌入比实数嵌入具有更强的归纳偏置，能以更少的参数实现更精确的表示。

当前的验证任务为 **Ring Span Inpainting**——在环形 token 序列中掩码连续片段，让模型预测缺失部分。这个任务看似简单，实则要求模型在嵌入空间中学习出完整的环拓扑结构。

---

## 架构

```
Input (token IDs)
  │
  ▼
ComplexEmbedding          ← 复数流形嵌入, Polar Adam 优化器
  │
  ▼
ComplexRotaryEmbedding    ← 复数旋转位置编码 (乘法, 非加法)
  │
  ▼
ComplexTransformerBlock   ← Pre-Norm 架构
  │  ├─ ComplexRMSNorm
  │  ├─ HolographicAttention    ← 共轭干涉 + MagPhaseSoftmax
  │  ├─ Residual
  │  ├─ ComplexRMSNorm
  │  ├─ FFN: ComplexLinear → PhaseTwist → ComplexLinear
  │  └─ Residual
  │
  ▼
Output (complex embeddings → nearest-neighbor decode)
```

### 核心组件

**HolographicAttention（全息共振注意力）**
不是 Query-Key Lookup，而是波干涉与输运系统。注意力分数 `S = Q @ K^H` 通过共轭转置产生复数干涉模式；MagPhaseSoftmax 将模长送入 softmax 做强度筛选，相位完整透传；输出 `O = A @ V` 实现相干输运，自动补偿相位延迟。

**PhaseTwist（双向幅相耦合激活）**
提供 AM→PM 和 PM→AM 的非线性耦合：`f(z) = m · exp(i(θ + γr))`，其中 `m = r(1 + β·cos(θ-φ))`。模长驱动相位旋转 (AM→PM)，相位调制模长增益 (PM→AM)。非线性完全来自三角函数而非截断操作。

**Polar Adam（极坐标 Adam 优化器）**
将 Wirtinger 梯度投影到参数的极坐标系，径向和切向分别维护独立的 Adam 一阶矩和二阶矩。这与架构中模长/相位的显式解耦保持一致——避免标准 Adam 中径向和切向共享自适应分母导致的相互干扰。

**Boltzmann-Elegant 损失函数（v4.4）**
将 L_Elegant 流形距离推广为 N 体 Boltzmann 分类。能量 `E_k = Σ_d [ln²(r/r̂) + |u-û|²]` 在径向和切向上解耦；自适应温度 `τ = std(E_competitors)` 仅基于竞争者能量计算，避免 target 能量干扰导致的温度失控；完整 Wirtinger 梯度含 τ-throughput 修正项。

---

## Config B 实验

Config B 是当前的核心容量压力测试配置。

### 任务定义

**Ring Span Inpainting**：V=256 个 token 排列在一个环上 (token i 的邻居是 token i±1 mod 256)。生成长度 32 的连续子序列，随机掩码 1~5 个连续位置，模型预测被掩码的 token。评估方式为最近邻解码——模型输出的复数向量与词表中所有 embedding 计算流形距离，取最近者作为预测。

### 配置参数

| 参数 | 值 | 说明 |
|------|----|------|
| V (词表) | 256 | 环上 token 数 |
| D (维度) | 64 | 复数嵌入维度 |
| V/D | 4 | 容量压力系数 |
| Heads | 4 | 注意力头数 (head_dim=16) |
| FFN | 256 | 前馈层维度 (4×D) |
| Params | 66,816 | 可学习参数数 (复数元素计) |
| Batch | 32 | 批大小 |
| LR | 0.001 | 学习率 |
| Epochs | 10,000 | 最大训练轮数 |
| Holdout | 20% | 测试集比例 (基于起始位置) |

### 实验结果 (v4.4)

| 指标 | Best Checkpoint | Final Model |
|------|----------------|-------------|
| Train Accuracy | 96.15% | — |
| Test Accuracy | 88.88% ± 2.70% | 89.40% ± 2.32% |
| NN% (环邻居一致率) | 44.9% | 55.5% |
| Self-Boltzmann p_target | 0.968 | — |
| Negative Margin Ratio | 0.0% | — |
| τ (最终) | 10.69 | — |
| p_target (最终) | 0.967 | — |
| EmbRMS | 0.736 | 0.681 |
| 训练时间 | 126s | — |

**稳定性诊断 (best → final)**：test_acc +0.52%, NN% +10.55%。最终模型未退化，训练全程保持稳定。

### 版本演进

**v4.3 → v4.4 的关键改进：**

| 维度 | v4.3 | v4.4 | 改进 |
|------|------|------|------|
| τ 稳定性 | 10→38 (失控飙升) | 9.8~11.4 (稳定) | 波动幅度缩小 17× |
| p_target | 0.6→0.17 (崩溃) | 持续上升至 0.97 | 不再崩溃 |
| neg% | 最终 24.8% | **0.0%** | 彻底消除 |
| 最终 Test Acc | 75.9% ± 6.7% | **88.9% ± 2.7%** | +13%, 方差↓ |
| 训练稳定性 | 后期剧烈震荡 | 全程平稳 | 本质改善 |

v4.4 的三个核心修改：

1. **τ 排除 target**：温度只度量竞争者（热浴）的能量分散度，不包含目标粒子。消除了"模型做对了反而被惩罚"的正反馈循环。
2. **force_b 含排斥力**：embedding 更新从纯吸引力升级为 Boltzmann Term 1 力（吸引 + 排斥），使 embedding 能直接推开竞争者。
3. **双重 Early Stopping**：accuracy AND loss 同时停滞才触发，避免在几何精进阶段过早终止。

---

## 项目结构

```
Anla/
├── core/
│   └── base_layer.py           # PolarAdamState, AdaptiveParamState, ComplexLayer 基类
├── layers/
│   ├── embedding.py            # ComplexEmbedding (稀疏 Polar Adam)
│   ├── positional.py           # ComplexRotaryEmbedding (乘法位置编码)
│   ├── normalization.py        # ComplexRMSNorm
│   ├── linear.py               # ComplexLinear (Polar Adam)
│   ├── activation.py           # PhaseTwist (AM-PM 双向耦合)
│   ├── holographic_attention.py # HolographicAttention + MagPhaseSoftmax
│   └── transformer_block.py    # ComplexTransformerBlock (Pre-Norm)
├── losses/
│   └── boltzmann_elegant.py    # Boltzmann-Elegant 损失 (v4.4)
├── experiments/
│   ├── capacity/
│   │   ├── capacity_pressure_test_v4.py  # 容量压力测试主训练脚本
│   │   └── run_config_B_train.py         # Config B 训练启动器
│   ├── ring/                   # Ring 实验
│   ├── torus/                  # Torus 实验
│   ├── manifold/               # 流形实验
│   ├── baselines/              # 基线对比实验
│   └── kernel/                 # 核方法实验
├── analysis/
│   ├── validate_tda_loops.py   # TDA 持续同调验证
│   ├── topology_audit_ring.py  # 环拓扑审计
│   └── extract_ring_embedding.py
├── visualization/
│   ├── visualize_config_B.py   # Config B 全量可视化 (10 张图 + summary)
│   └── visualize_ring_masking.py
├── diagnostics/
│   ├── diagnostic_probe.py     # 诊断探针
│   ├── gradient_probe.py       # 梯度探针
│   └── wirtinger_gradcheck.py  # Wirtinger 梯度验证
├── tests/                      # 单元测试
├── utils/
│   ├── complex_ops.py          # 复数工具函数
│   └── torus_data.py           # 环面数据生成
├── tools/
│   └── split_for_ai.py         # 代码分割工具
├── docs/
│   └── theory_complex_vs_real.md  # 复数 vs 实数网络理论分析
└── Logs/                       # 实验日志与可视化输出
```

---

## 快速开始

### 环境要求

```
Python >= 3.8
PyTorch >= 2.0
NumPy, SciPy, Matplotlib
ripser (可选, TDA 分析)
```

### 运行 Config B 训练

```bash
python -m Anla.experiments.capacity.run_config_B_train
```

输出目录：`Logs/config_B_analysis/config_B_v256_d64/`

### 运行可视化分析

```bash
python -m Anla.visualization.visualize_config_B
```

生成 10 张诊断图和量化 summary report：

| 图号 | 内容 | 分析维度 |
|------|------|----------|
| 01 | Training Dynamics | Loss, Accuracy, ρ, EmbRMS, 泛化 gap |
| 02 | Boltzmann Diagnostics | τ, p_target, neg%, energy gap |
| 03 | Embedding PCA | 2D/3D 主成分投影 |
| 04 | Phase Structure | 相位图, 相位差, 线性度, unwrap |
| 05 | Ring Topology | NN 分布, ring/embedding 距离相关性 |
| 06 | Distance Matrix | 全矩阵, 局部放大, 配对分布 |
| 07 | Energy Landscape | target vs wrong, gap 分布, Self-Boltzmann |
| 08 | Magnitude Analysis | per-token/per-dim RMS, 相位-模长联合分布 |
| 09 | TDA Persistence | H0/H1 持续图, barcode, dominance ranking |
| 10 | Attention Pattern | 4 头注意力模长与相位 |

v4.4 支持双轨分析模式：当存在 `ring_features_best.npz` 和 `ring_features_final.npz` 时，自动对 best checkpoint 和 final model 分别生成全套可视化，并在 summary report 中输出退化幅度诊断。

---

## 数学基础

### L_Elegant 流形距离

对于预测向量 z 与目标嵌入 ẑ_k，逐维度计算径向和切向距离后求和：

```
E_k(z) = Σ_d [ ln²(|z_d| / |ẑ_{k,d}|) + |z_d/|z_d| - ẑ_{k,d}/|ẑ_{k,d}||² ]
```

径向项 `ln²(r/r̂)` 衡量模长在对数尺度上的偏差；切向项 `|u - û|²` 衡量单位相位向量的偏差。两项解耦处理，与架构中的极坐标设计一致。

### Boltzmann 分类 (v4.4)

将能量转化为分类概率：

```
τ_n = std({E_k : k ≠ target_n})
p_k = softmax(-E_k / τ_n)
L = -log p_target
```

温度 τ 仅基于竞争者能量的标准差计算（排除 target），保证模型学好 target 后 τ 不会失控。等价于 softmax 输入的自适应标准化。

### Wirtinger 梯度

所有反向传播基于 Wirtinger 微积分。对于实值损失 L 和复数参数 w：

```
dL/dw* = (1/2)(∂L/∂Re(w) + i·∂L/∂Im(w))
```

每一层的 `manual_backward` 方法实现了精确的 Wirtinger 链式法则，经过 `wirtinger_gradcheck.py` 与有限差分法交叉验证。

---

## 已知限制与后续方向

**当前限制：**

- **单层 Transformer**：当前模型仅使用一个 TransformerBlock，深度有限。
- **频率覆盖 50%**：D=64 维最多编码 64 个频率，而 V=256 的环需要 128 个频率才能完美表示。模型需要通过非线性组合弥补这一理论缺口。
- **相位结构缺失**：v4.4 的排斥力使模型转向高维散点编码，丧失了可解释的频率编码结构。
- **EmbRMS 缓慢下降**：embedding 模长从 1.0 缓慢降至 0.69，趋势虽已大幅减缓但未完全消除。

**后续方向：**

- 拓扑保持排斥力（仅推开 ring distance > k 的 token，保护近邻聚集）
- 多层 Transformer 堆叠与层间梯度流分析
- 更高压力配置（V/D=8, 16）的容量边界探索
- 实数基线 Transformer 的等参数量对比实验

---

## 许可

本项目为个人研究项目。
