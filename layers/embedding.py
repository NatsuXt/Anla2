"""
保存位置: Anla/layers/embedding.py

ComplexEmbedding — 复数流形嵌入层
=========================================================================

v5.0 → v5.1 变更:
    [1] 撤销 project_out_radial
        v5.0 在 Path A 中使用 project_out_radial=True, 封锁了径向更新。
        实际训练中发现: 当 Decoder/Encoder 有 weight_decay 时,
        z_pred 模长系统性缩小, Path B 跟随将 embedding 模长拉向 0,
        而 Path A 被封锁无法提供恢复力, 导致 EmbRMS 不可逆崩溃。

        v5.1 的修复策略:
            a) Decoder/Encoder 移除 weight_decay (消除根源)
            b) 撤销 project_out_radial (恢复 Path A 的全量梯度)
        
        理论依据:
            在 Encoder-Decoder 架构中, Path A 的梯度经过 Encoder Linear
            的反向传播后, RMSNorm 的径向偏置已被矩阵乘法混合,
            不再是纯径向信号。project_out_radial 投影掉的是混合梯度
            中的径向分量, 而非"RMSNorm 偏置"。

v4 → v5.0 变更 (保留):
    PolarAdamState 仍保留 project_out_radial 功能 (base_layer.py),
    供未来实验使用。但当前默认不启用。

    其余与 v4 完全一致。
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
    3. [v5] Path A 径向投影 (project_out_radial)

    v5: Path A 移除径向梯度, 防止 RMSNorm 导致的 EmbRMS 漂移。
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
        """
        Path A: 来自 Transformer 反向传播的梯度更新。

        [v5.1] 撤销 project_out_radial:
            v5.0 中使用 project_out_radial=True 封锁了 Path A 的径向更新,
            理由是 RMSNorm 反向传播产生系统性径向偏置。

            但在 v5 的 Encoder-Decoder 架构中, Path A 的梯度在到达
            Embedding 之前经过了 Encoder Linear 的反向传播
            (grad_input = grad_output @ W^*), 这个矩阵乘法会将
            径向和切向混合。RMSNorm 的纯径向偏置经过 Encoder Linear
            后不再是纯径向的。project_out_radial 投影掉的不再是
            "偏置", 而是包含有效信息的混合梯度。

            同时, Decoder 和 Encoder 已移除 weight_decay (v5.1),
            消除了 z_pred 模长系统性缩小的根源。在此条件下,
            Path A 的径向梯度不再携带单方向的偏置信号。

            恢复全量梯度, 让 Path A 和 Path B 共同控制模长。
        """
        if self.input_cache is None:
            return
        # [v5.1] 恢复全量梯度 (project_out_radial=False)
        self._apply_update(grad_output, self.input_cache, lr, weight_decay,
                           project_out_radial=False)
        self.input_cache = None

    def manual_backward_explicit(self, grad: torch.Tensor,
                                 indices: torch.Tensor,
                                 lr: float, weight_decay: float = 0.0):
        """
        Path B: [双向纠缠接口]

        Path B 不经过 Norm 层, 径向信号来自 L_Elegant,
        是有物理意义的。保持全量梯度 (project_out_radial=False)。
        """
        # [v5] Path B: 保留全量梯度 (包括径向)
        self._apply_update(grad, indices, lr, weight_decay,
                           project_out_radial=False)

    def _apply_update(self, grad: torch.Tensor, indices: torch.Tensor,
                      lr: float, weight_decay: float,
                      project_out_radial: bool = False):
        """
        核心动力学更新逻辑。

        v5 变更: 新增 project_out_radial 参数, 传递给
                 PolarAdamState.sparse_step。

        流程:
            1. 维度展平 + 梯度聚合 (同一 token 多次出现时求和)
            2. PolarAdamState.sparse_step:
               a. [v5] 如果 project_out_radial=True, 先移除径向分量
               b. 极坐标分解 (h, t)
               c. 独立 Adam 矩更新
               d. 偏差校正
               e. 合成复数更新
               f. 回写参数和状态
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
        #    [v5] project_out_radial 传递给 sparse_step
        with torch.no_grad():
            self._optim.sparse_step(
                self.weight, unique_ids, grad_sum,
                lr, weight_decay,
                project_out_radial=project_out_radial,
            )
