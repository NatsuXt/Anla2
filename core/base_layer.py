"""
保存位置: Anla/core/base_layer.py

v4 — 极坐标 Adam 优化器 (Polar Adam)
=========================================================================

变更概要:
    1. AdaptiveParamState: RMSProp → 完整 Adam (一阶矩 + 偏差校正)
       用于实数参数 (PhaseTwist 的 γ/β/φ, RMSNorm 的 scale)

    2. [新增] PolarAdamState: 为复数参数设计的极坐标分解 Adam
       用于 ComplexLinear 和 ComplexEmbedding

设计动机:
    Anla 的计算流在多处显式区分模长和相位:
    · MagPhaseSoftmax: 模长走 Softmax, 相位直传
    · L_Elegant: log(r/r̂)² + |u-û|² 模长相位解耦
    · PhaseTwist: AM-PM / PM-AM 双向耦合

    标准复数 Adam 用 |g|² 做自适应分母, 径向和切向共享归一化 —
    这与架构的极坐标设计思想矛盾。

    Polar Adam 将梯度投影到参数的极坐标系:
        h = Re(g · e^{-iφ})  (径向, 控制模长)
        t = Im(g · e^{-iφ})  (切向, 控制相位)
    然后分别维护独立的一阶矩和二阶矩, 使径向和切向
    拥有独立的自适应学习率, 互不干扰。

    径向阻尼 (radial_scale):
        Embedding 层的特殊需求 — 前向路径中 RMSNorm/Softmax/tanh
        均为模长压缩操作, 反向梯度穿过这些层时携带系统性的
        "让输入更小"信号。动量累积此一致信号, 导致 EmbRMS 持续下降。
        
        radial_scale < 1.0 降低径向步长, 让 L_Elegant 的径向力
        (而非噪声梯度流) 主导模长变化。相位学习率不受影响。
        
        ComplexLinear 默认 radial_scale=1.0 (权重模长编码增益, 需全速适应)。
        ComplexEmbedding 默认 radial_scale=1.0 (v4 移除 tanh 后不再需要径向阻尼)。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# =====================================================================
#  实数参数优化器: Adam (升级自 v2 的 RMSProp)
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

    核心思想:
        将复数 Wirtinger 梯度 g = dL/dw* 分解为:
            径向 h = Re(g · e^{-iφ})  → 调节模长 (权重"多强")
            切向 t = Im(g · e^{-iφ})  → 旋转相位 (权重"何向")

        分别维护独立的一阶矩 (m_r, m_t) 和二阶矩 (v_r, v_t),
        使径向和切向拥有独立的自适应学习率。

    与标准复数 Adam 的区别:
        标准方法: v = EMA(|g|²), step = g / √v
            → 径向和切向共享分母, 大切向梯度会压制径向学习
        本方法: v_r = EMA(h²), v_t = EMA(t²), 分别归一化
            → 两个方向互不干扰

    数学推导:
        设 w = |w|·e^{iφ}, g = dL/dw* (Wirtinger 共轭梯度)

        Step 1 — 极坐标分解:
            e = e^{iφ} = w / |w|         (当前权重的单位相位向量)
            h = Re(g · conj(e))           (径向投影, 实数)
            t = Im(g · conj(e))           (切向投影, 实数)

        Step 2 — 独立 Adam:
            m_r = β₁·m_r + (1-β₁)·h,     v_r = β₂·v_r + (1-β₂)·h²
            m_t = β₁·m_t + (1-β₁)·t,     v_t = β₂·v_t + (1-β₂)·t²

        Step 3 — 偏差校正:
            m̂_r = m_r/(1-β₁^k), v̂_r = v_r/(1-β₂^k), ...

        Step 4 — 合成复数更新:
            step_r = m̂_r / (√v̂_r + ε)   (径向步长, 实数)
            step_t = m̂_t / (√v̂_t + ε)   (切向步长, 实数)
            Δw = e · (step_r + i·step_t)  (复数更新向量)

        Step 5 — 应用:
            w -= lr · Δw

    超参数选择:
        β₁ = 0.9:  一阶矩衰减, 平滑 ~10 步 (标准 Adam)
        β₂ = 0.99: 二阶矩衰减, 追踪 ~100 步 (标准 Adam)
        eps = 1e-8: 分母安全值 (标准 Adam)
        radial_scale = 1.0: 径向步长缩放因子

    径向阻尼 (radial_scale < 1):
        Embedding 的语义信息主要编码在相位 (经 RMSNorm 后模长被归一化),
        模长是次要自由度。设 radial_scale=0.1 使:
          · 相位以全速学习 (语义载体)
          · 模长以 1/10 速度学习 (减缓压缩层梯度流导致的系统性漂移)

        原理: 前向路径中 RMSNorm / Softmax / tanh 均为模长压缩操作,
              反向梯度穿过这些层时携带系统性的"让输入更小"信号。
              动量 (β₁) 累积此一致信号, 导致模长持续下降。
              径向阻尼不与梯度方向冲突 (不像 RMS 稳态做硬投影),
              仅降低响应速度, 让 L_Elegant 的径向力有足够时间主导。

        ComplexLinear 使用默认 radial_scale=1.0 (无阻尼),
        因为线性层权重的模长同时编码增益信息, 需要全速适应。
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

        Args:
            param:        nn.Parameter (complex dtype)
            complex_grad: Wirtinger 梯度 dL/dw*, 与 param 同 shape
            lr:           学习率
            weight_decay: 解耦权重衰减 (仅收缩模长, 不改变相位)
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
                    lr: float, weight_decay: float = 0.0):
        """
        稀疏版本 — 仅更新 indices 指定的行。
        用于 Embedding 层 (每步只有活跃 token 被更新)。

        Args:
            param:        nn.Parameter (complex, shape [V, D])
            indices:      活跃行索引 (1D LongTensor)
            complex_grad: 仅活跃行的梯度 (shape [len(indices), D])
            lr:           学习率
            weight_decay: 解耦权重衰减
        """
        with torch.no_grad():
            self.step_count += 1

            # 1. 提取活跃行
            w = param.data[indices]
            mag_w = w.abs().clamp(min=1e-7)
            e = w / mag_w

            projected = complex_grad * e.conj()
            h = projected.real
            t = projected.imag

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
#  基类
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
