"""
保存位置: Anla/core/base_layer.py

v5 — 极坐标 Adam 优化器 (Polar Adam) + 径向投影
=========================================================================

v4 → v5 变更:
    [1] PolarAdamState.sparse_step 新增 project_out_radial 参数
    
        动机 (来自理论分析):
            Path A (Transformer 反向传播 → Embedding) 穿过 RMSNorm 层。
            RMSNorm 前向消除了输入模长信息, 其反向传播的链式法则
            产生系统性的"收缩"信号 — 这是 RMSNorm 的数学副产物,
            不携带任何关于任务的信息。

            具体来说, RMSNorm 反向传播公式:
                dL/dz = (1/RMS) · (dL/dẑ - ẑ · Re<dL/dẑ, ẑ>/D)
            其中第二项 -ẑ·(.../D) 系统性地移除梯度的径向分量,
            并在残差连接中引入指向收缩方向的偏置。

            Path B (Boltzmann 力 → Embedding) 不经过任何 Norm 层。
            L_Elegant 的径向力 ln(r/r̂)/r · u 直接度量模长差异,
            是有物理意义的径向信号。

        修正:
            project_out_radial=True 时, 在投影到极坐标之前,
            将梯度的径向分量移除:
                g_tangential = g - Re(g · conj(e)) · e
            使 Path A 只更新 Embedding 的相位, 不影响模长。
            Path B 保持全量梯度, 模长由 Boltzmann 力场控制。

        效果:
            · EmbRMS 漂移消失 (径向偏置被移除)
            · 模长均衡由 L_Elegant 径向力决定 (Path B)
            · 相位学习不受影响 (切向分量完整保留)

    其余 v4 代码完全不变。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# =====================================================================
#  实数参数优化器: Adam (与 v4 完全一致)
# =====================================================================
class AdaptiveParamState:
    """
    为实数参数提供完整的 Adam 优化器。

    v4 升级 (相对于 v2 RMSProp):
        [+] 一阶矩 m (动量, β₁=0.9)
        [+] 偏差校正 (消除冷启动偏差)
        [Δ] β₂: 0.90 → 0.99 (与标准 Adam 一致, 更稳定)
        [Δ] eps: 1e-5 → 1e-8 (标准 Adam 精度)
        [-] init_energy: 不再需要 (偏差校正替代冷启动保护)

    接口不变: PhaseTwist / RMSNorm 无需修改调用代码。
    """

    def __init__(self, shape: tuple, device: torch.device = None,
                 beta1: float = 0.9, beta2: float = 0.99,
                 eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # 一阶矩 (动量)
        self.m = torch.zeros(shape, dtype=torch.float32, device=device)
        # 二阶矩 (自适应学习率)
        self.v = torch.zeros(shape, dtype=torch.float32, device=device)
        # 步数计数器 (偏差校正)
        self.step_count = 0

    def to(self, device: torch.device):
        self.m = self.m.to(device)
        self.v = self.v.to(device)
        return self

    def step(self, param: nn.Parameter, raw_grad: torch.Tensor,
             lr: float, weight_decay: float = 0.0):
        """
        执行一步 Adam 更新。

        接口与 v2 完全一致, 内部逻辑升级为:
            m = β₁·m + (1-β₁)·g
            v = β₂·v + (1-β₂)·g²
            m̂ = m / (1-β₁^t),  v̂ = v / (1-β₂^t)
            param -= lr · m̂ / (√v̂ + ε)
        """
        with torch.no_grad():
            self.step_count += 1

            # 实数梯度
            g = raw_grad

            # 更新矩
            self.m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v.mul_(self.beta2).add_(g.pow(2), alpha=1.0 - self.beta2)

            # 偏差校正
            bc1 = 1.0 - self.beta1 ** self.step_count
            bc2 = 1.0 - self.beta2 ** self.step_count
            m_hat = self.m / bc1
            v_hat = self.v / bc2

            # 自适应步长
            adaptive_step = m_hat / (v_hat.sqrt() + self.eps) * lr

            # 解耦权重衰减
            if weight_decay > 0:
                param.data.mul_(1.0 - weight_decay)

            param.data.sub_(adaptive_step)


# =====================================================================
#  复数参数优化器: Polar Adam (极坐标 Adam)
# =====================================================================
class PolarAdamState:
    """
    极坐标 Adam — 为复数参数设计的自适应优化器。

    v5 新增:
        project_out_radial 参数 (sparse_step 方法):
            True  — 在极坐标分解前移除梯度的径向分量,
                    仅保留切向 (相位) 更新。
                    用于 Path A (Transformer → Embedding),
                    防止 RMSNorm 反向传播的系统性径向偏置。
            False — 保持全量梯度 (默认, 与 v4 行为一致)。
                    用于 Path B (Boltzmann → Embedding)。

    其余与 v4 完全一致。
    """

    def __init__(self, shape: tuple, device: torch.device = None,
                 beta1: float = 0.9, beta2: float = 0.99,
                 eps: float = 1e-8, radial_scale: float = 1.0):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.radial_scale = radial_scale

        # 径向矩 (Radial moments)
        self.m_r = torch.zeros(shape, dtype=torch.float32, device=device)
        self.v_r = torch.zeros(shape, dtype=torch.float32, device=device)

        # 切向矩 (Tangential moments)
        self.m_t = torch.zeros(shape, dtype=torch.float32, device=device)
        self.v_t = torch.zeros(shape, dtype=torch.float32, device=device)

        # 步数计数器
        self.step_count = 0

    def to(self, device: torch.device):
        self.m_r = self.m_r.to(device)
        self.v_r = self.v_r.to(device)
        self.m_t = self.m_t.to(device)
        self.v_t = self.v_t.to(device)
        return self

    def _compute_polar_step(self, h: torch.Tensor, t: torch.Tensor,
                            mr: torch.Tensor, mt: torch.Tensor,
                            vr: torch.Tensor, vt: torch.Tensor):
        """
        内部工具: 给定径向/切向梯度和对应状态,
        执行 Adam 矩更新并返回偏差校正后的自适应步长。

        返回: (step_r, step_t) 均为实数张量
        """
        # 更新矩
        mr.mul_(self.beta1).add_(h, alpha=1.0 - self.beta1)
        mt.mul_(self.beta1).add_(t, alpha=1.0 - self.beta1)
        vr.mul_(self.beta2).add_(h.pow(2), alpha=1.0 - self.beta2)
        vt.mul_(self.beta2).add_(t.pow(2), alpha=1.0 - self.beta2)

        # 偏差校正
        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count
        m_r_hat = mr / bc1
        m_t_hat = mt / bc1
        v_r_hat = vr / bc2
        v_t_hat = vt / bc2

        # 自适应步长
        step_r = m_r_hat / (v_r_hat.sqrt() + self.eps)
        step_t = m_t_hat / (v_t_hat.sqrt() + self.eps)

        # 径向阻尼: 降低模长变化速率
        if self.radial_scale != 1.0:
            step_r = step_r * self.radial_scale

        return step_r, step_t

    def step(self, param: nn.Parameter, complex_grad: torch.Tensor,
             lr: float, weight_decay: float = 0.0):
        """
        对整个复数参数张量执行一步 Polar Adam 更新。
        与 v4 完全一致。
        """
        with torch.no_grad():
            self.step_count += 1

            # 1. 极坐标分解
            w = param.data
            mag_w = w.abs().clamp(min=1e-7)
            e = w / mag_w                       # 单位相位向量

            projected = complex_grad * e.conj()  # 投影到极坐标
            h = projected.real                    # 径向分量
            t = projected.imag                    # 切向分量

            # 2. Adam 更新 (径向/切向独立)
            step_r, step_t = self._compute_polar_step(
                h, t, self.m_r, self.m_t, self.v_r, self.v_t
            )

            # 3. 解耦权重衰减 (仅收缩模长)
            if weight_decay > 0:
                param.data.mul_(1.0 - weight_decay)

            # 4. 合成复数更新并应用
            complex_step = e * (step_r + 1j * step_t)
            param.data.sub_(complex_step * lr)

    def sparse_step(self, param: nn.Parameter,
                    indices: torch.Tensor,
                    complex_grad: torch.Tensor,
                    lr: float, weight_decay: float = 0.0,
                    project_out_radial: bool = False):
        """
        稀疏版本 — 仅更新 indices 指定的行。
        用于 Embedding 层 (每步只有活跃 token 被更新)。

        Args:
            param:        nn.Parameter (complex, shape [V, D])
            indices:      活跃行索引 (1D LongTensor)
            complex_grad: 仅活跃行的梯度 (shape [len(indices), D])
            lr:           学习率
            weight_decay: 解耦权重衰减
            project_out_radial: [v5 新增]
                True  — 移除梯度的径向分量, 仅更新相位。
                         用于 Path A, 阻止 RMSNorm 的径向偏置
                         传入 Embedding 模长。
                False — 保持全量梯度 (默认, v4 行为)。
                         用于 Path B, 让 Boltzmann 力场
                         完全控制 Embedding 模长。

        [v5] project_out_radial 的实现:
            在极坐标分解之前, 将梯度投影到当前权重的切平面:
                e = w / |w|                        (单位相位向量)
                h_raw = Re(g · conj(e))            (径向分量)
                g_tangential = g - h_raw · e       (移除径向, 保留切向)
            
            之后的 Polar Adam 步骤正常进行, 但由于径向分量已被
            移除, h = Re(g_tangential · conj(e)) ≈ 0。
            实际上我们直接将 h 置零, 避免浮点误差累积。
        """
        with torch.no_grad():
            self.step_count += 1

            # 1. 提取活跃行
            w = param.data[indices]
            mag_w = w.abs().clamp(min=1e-7)
            e = w / mag_w

            # [v5] 径向投影: 移除梯度的径向分量
            if project_out_radial:
                # 计算径向投影 (实数)
                h_raw = (complex_grad * e.conj()).real
                # 从梯度中减去径向分量, 只保留切向
                complex_grad = complex_grad - h_raw * e

            projected = complex_grad * e.conj()
            h = projected.real
            t = projected.imag

            # [v5] 如果已经投影掉径向, h 理论上为 0
            # 但浮点运算可能残留微小值, 显式置零以保证干净
            if project_out_radial:
                h = torch.zeros_like(h)

            # 2. 提取活跃行的优化器状态
            mr = self.m_r[indices]
            mt = self.m_t[indices]
            vr = self.v_r[indices]
            vt = self.v_t[indices]

            # 3. Adam 更新
            step_r, step_t = self._compute_polar_step(
                h, t, mr, mt, vr, vt
            )

            # 4. 权重衰减
            if weight_decay > 0:
                w.mul_(1.0 - weight_decay)

            # 5. 合成更新
            complex_step = e * (step_r + 1j * step_t)
            w.sub_(complex_step * lr)

            # 6. 回写参数和状态
            param.data.index_put_((indices,), w)
            self.m_r.index_put_((indices,), mr)
            self.m_t.index_put_((indices,), mt)
            self.v_r.index_put_((indices,), vr)
            self.v_t.index_put_((indices,), vt)


# =====================================================================
#  基类 (与 v4 完全一致)
# =====================================================================
class ComplexLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.input_cache = None
        self.output_cache = None

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        pass

    def clear_cache(self):
        self.input_cache = None
        self.output_cache = None
