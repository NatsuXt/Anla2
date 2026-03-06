"""
保存位置: Anla/layers/embedding.py

ComplexEmbedding — 复数流形嵌入层
=========================================================================

v4 变更:
    优化器: 内联 RMSProp → Polar Adam (极坐标 Adam, 稀疏版)

    原版问题 (与 ComplexLinear 相同):
        · |g|² 单体追踪: 径向和切向共享自适应分母
        · 无动量: 稀疏更新的 token 缺少跨步记忆
        · 无偏差校正: init_energy=1e-3 是 ad-hoc 的

    修正:
        · 极坐标分解 + 独立 Adam
        · 稀疏更新: 仅修改活跃 token 的参数和优化器状态
        · 全局步数计数器: 偏差校正基于训练步数而非 token 出现次数
          (与标准 Adam 处理 Embedding 的方式一致)

    v3 → v4 保留:
        · weight_decay=0.0 (由调用方保证, Embedding 不施加衰减)
        · RMS 稳态已删除 (v3 决定), 模长由 L_Elegant 径向力约束
        · Path B 双向纠缠接口保留

    v3 注意: 虽然 v4 的 PolarAdamState 内部包含 weight_decay 支持,
            但调用方 (train_ring_masking.py 和 capacity_pressure_test.py)
            已经在 v3 中将 embedding 的 weight_decay 设为 0.0。
            这个决定在 v4 中不变。
"""

import torch
import torch.nn as nn
from Anla.core.base_layer import ComplexLayer, PolarAdamState
from Anla.utils.complex_ops import complex_kaiming_normal_


class ComplexEmbedding(ComplexLayer):
    """
    [Anla AGI Core] Manifold Embedding Layer
    支持：
    1. Polar Adam 优化器 (极坐标自适应, 径向/切向独立)
    2. 双向纠缠接口 (Bidirectional Entanglement Interface)

    v4: RMSProp → Polar Adam
    v3: 删除 RMS 稳态, 模长约束交由 L_Elegant 径向力。
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 1. 复数流形初始化
        weight_real = torch.empty(num_embeddings, embedding_dim)
        weight_imag = torch.empty(num_embeddings, embedding_dim)
        complex_kaiming_normal_(weight_real, weight_imag)

        # 初始化归一化：保证起点在单位球体附近
        raw_w = torch.complex(weight_real, weight_imag)
        init_w = raw_w / (raw_w.abs() + 1e-9)
        self.weight = nn.Parameter(init_w)

        # 2. [v4] Polar Adam 优化器状态
        self._optim = PolarAdamState(
            shape=(num_embeddings, embedding_dim)
        )

        self.input_cache = None

    def _ensure_optim_device(self):
        """确保优化器状态与参数在同一设备上。"""
        self._optim.to(self.weight.device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.input_cache = input_ids if self.training else None
        with torch.no_grad():
            return nn.functional.embedding(input_ids, self.weight)

    def manual_backward(self, grad_output: torch.Tensor,
                        lr: float, weight_decay: float = 0.0):
        """Path A: 针对 forward 输入的默认反向传播"""
        if self.input_cache is None:
            return
        self._apply_update(grad_output, self.input_cache, lr, weight_decay)
        self.input_cache = None

    def manual_backward_explicit(self, grad: torch.Tensor,
                                 indices: torch.Tensor,
                                 lr: float, weight_decay: float = 0.0):
        """Path B: [双向纠缠接口]"""
        self._apply_update(grad, indices, lr, weight_decay)

    def _apply_update(self, grad: torch.Tensor, indices: torch.Tensor,
                      lr: float, weight_decay: float):
        """
        核心动力学更新逻辑。

        v4 变更: RMSProp → Polar Adam (稀疏版)
            · 梯度聚合后, 投影到每个权重的极坐标系
            · 径向/切向分别维护一阶矩和二阶矩
            · 仅更新活跃 token 的参数和优化器状态

        流程:
            1. 维度展平 + 梯度聚合 (同一 token 多次出现时求和)
            2. PolarAdamState.sparse_step:
               a. 极坐标分解 (h, t)
               b. 独立 Adam 矩更新
               c. 偏差校正
               d. 合成复数更新
               e. 回写参数和状态
        """
        self._ensure_optim_device()

        # 1. 维度展平
        grad_flat = grad.reshape(-1, self.embedding_dim)
        ids_flat = indices.reshape(-1)

        # 2. 梯度聚合 (同一 token 在 batch 中多次出现时, 梯度求和)
        unique_ids, inverse_indices = torch.unique(ids_flat, return_inverse=True)

        grad_sum = torch.zeros(len(unique_ids), self.embedding_dim,
                               dtype=grad_flat.dtype, device=grad_flat.device)
        grad_sum.index_add_(0, inverse_indices, grad_flat)

        # 3. Polar Adam 稀疏更新
        with torch.no_grad():
            self._optim.sparse_step(
                self.weight, unique_ids, grad_sum,
                lr, weight_decay
            )
