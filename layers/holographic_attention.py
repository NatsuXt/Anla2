"""
保存位置: Anla/layers/holographic_attention.py

全息共振注意力 (Holographic Resonance Attention)
=========================================================================

修正说明 (相对于 v4):

[Fix #1] MagPhaseSoftmax.manual_backward — 切向梯度数值稳定性
    
    问题:
        当 |S_j| ≈ 0 时 (训练初期 Q 与 K 近正交的常见情况),
        切向梯度 tangential_grad = p_j · t_j / |S_j| 中的分母趋近零,
        导致梯度数值爆炸。
    
        物理上, |S_j| ≈ 0 意味着 angle(S_j) 未定义 — 相位是随机的。
        对不确定的相位施加大梯度, 等于向 Q/K 投影层注入纯噪声。
    
        后果链:
            切向爆炸 → Q/K 权重被噪声更新 → Attention 模式不稳定
            → ρ 快速升到 0.3~0.4 后停滞 → 下游所有层收到嘈杂信号
    
    修正方法:
        引入平滑抑制函数 w(|S|), 使切向梯度在 |S| → 0 时自动衰减为零:
    
            tangential_grad_j = w(|S_j|) · p_j · t_j / (|S_j| + ε)
    
        其中:
            w(r) = r² / (r² + τ²)
    
        性质:
            · 当 |S_j| >> τ 时: w ≈ 1, 与原版完全一致
            · 当 |S_j| << τ 时: w ≈ |S_j|²/τ², 切向力线性趋零
            · 综合效果: tangential_grad ∝ |S_j|² / (|S_j|² + τ²) · 1/|S_j|
                        = |S_j| / (|S_j|² + τ²)
              这在 |S_j|=0 时为 0, 在 |S_j|=τ 时达到峰值 1/(2τ),
              此后缓慢衰减为 1/|S_j| — 完美过渡。
    
        τ 的选择:
            τ = 0.01 — 当 |S_j| > 0.1 (约 10τ) 时 w > 0.99, 几乎无影响。
            只在 |S_j| < 0.01 的"相位模糊区"内起保护作用。

    数学验证:
        此修正不改变 Wirtinger 导数的解析公式 — 它等价于将 tangential
        贡献乘以一个与上游梯度无关的标量权重。当 |S| 远离零时, 权重
        为 1, 修正量为零阶小量。因此在 |S| 有限处, 梯度验证结果不受影响。

=========================================================================

以下部分与 v4 完全一致 (MagPhaseSoftmax.forward, HolographicAttention 整体):

[v4 Critical Fix] MagPhaseSoftmax.manual_backward 完全重写 (保留)
    
    修正方法 (保留):
    从 Wirtinger 微积分严格推导 dL/dS*，最终公式为:
    
        dL/dS_j* = e_j · { scale · JVP_softmax(h)_j  +  i · p_j · t_j / |S_j| }
    
    其中:
        e_j = e^{iθ_j}                         (输入 S_j 的单位相位向量)
        p_j = softmax_j(scale · |S|)            (前向输出的模长/概率)
        h_k = Re(G_k · e^{-iθ_k})              (上游梯度在径向的投影)
        t_j = Im(G_j · e^{-iθ_j})              (上游梯度在切向的投影)
        JVP_softmax(h)_j = p_j · (h_j - Σ_k p_k h_k)
                                                (标准 Softmax Jacobian-向量积)

    完整推导:
        前向: A_k = p_k · e^{iθ_k}
        
        Step 1 — 极坐标偏导:
            ∂A_k/∂|S_j| = (∂p_k/∂|S_j|) · e_k
                         = scale · p_k(δ_{kj} - p_j) · e_k
            ∂A_k/∂θ_j   = δ_{kj} · i · A_k
        
        Step 2 — Wirtinger 变换 (|S|, θ) → (S, S*):
            ∂|S_j|/∂S_j  = e_j*/2          ∂|S_j|/∂S_j*  = e_j/2
            ∂θ_j/∂S_j    = -ie_j*/(2|S_j|) ∂θ_j/∂S_j*    = ie_j/(2|S_j|)
        
        Step 3 — 链式法则合并:
            ∂A_k/∂S_j* = scale · p_k(δ_{kj}-p_j) · e_j · e_k / 2
                        + δ_{kj} · i · A_k · ie_j/(2|S_j|)
                        = scale · p_k(δ_{kj}-p_j) · e_j · e_k / 2
                        - δ_{kj} · p_k · e_k² / (2|S_k|)       ... (*)
            
            ∂A_k/∂S_j  = scale · p_k(δ_{kj}-p_j) · e_j* · e_k / 2
                        + δ_{kj} · p_k / (2|S_k|)               ... (**)
        
        Step 4 — Wirtinger 链式法则 (G = dL/dA*, 实值 L):
            dL/dS_j* = Σ_k [ conj(G_k) · (*) + G_k · conj(**) ]
            
            径向部分 (来自 Softmax):
                Σ_k [conj(G_k)·scale·p_k(δ_{kj}-p_j)·e_j·e_k/2
                    + G_k·scale·p_k(δ_{kj}-p_j)·e_j·conj(e_k)/2]
                = scale/2 · e_j · Σ_k p_k(δ_{kj}-p_j) · [conj(G_k)·e_k + G_k·conj(e_k)]
                = scale/2 · e_j · Σ_k p_k(δ_{kj}-p_j) · 2·Re(G_k · e_k*)
                = scale · e_j · Σ_k p_k(δ_{kj}-p_j) · h_k
                = scale · e_j · JVP_softmax(h)_j
                
                注: 1/2 因子与 2·Re(...) 的 2 精确抵消
            
            切向部分 (来自相位, 仅 k=j 贡献):
                conj(G_j) · (-p_j·e_j²/(2|S_j|)) + G_j · conj(p_j/(2|S_j|))
                = p_j/(2|S_j|) · [G_j - conj(G_j)·e_j²]

                令 G_j = a·e_j + b·i·e_j，则:
                    G_j - conj(G_j)·e_j² = (a+bi)·e_j - (a-bi)·e_j = 2bi·e_j
                    其中 b = Im(G_j · conj(e_j)) = t_j

                = p_j/(2|S_j|) · 2i · t_j · e_j
                = i · p_j · t_j · e_j / |S_j|
                
                注: 1/2 因子与虚部展开的 2i 精确抵消

=========================================================================
"""

import torch
import torch.nn as nn
import math
from Anla.core.base_layer import ComplexLayer
from Anla.layers.linear import ComplexLinear


class MagPhaseSoftmax(ComplexLayer):
    """
    [核心算子] 幅相分离归一化 (Magnitude-Phase Decoupled Softmax)
    
    物理意义:
    1. 模长(Magnitude): 代表波的相干强度，经过 Softmax 筛选显著信号。
    2. 相位(Phase): 代表波的传播方向，完全保留，不做任何改变。
    
    Forward:
        A = softmax(|S| · scale) · e^{i · angle(S)}
        
    Manual Backward:
        精确 Wirtinger 导数, 含切向梯度平滑抑制 (Fix #1)。
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.softmax_mag_cache = None
        self.phase_cache = None
        self.scale_factor = None

    def forward(self, s: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        前向传播: 幅相分离 → Softmax 模长筛选 → 全息重组。

        Args:
            s: 复数注意力分数矩阵, shape (Batch, Heads, Seq_Q, Seq_K)
            scale: 模长缩放因子 (通常为 1/sqrt(d_head))

        Returns:
            复数注意力权重矩阵 A = softmax(|s|·scale) · e^{i·angle(s)}
        """
        # 缓存原始输入 (backward 需要 |S| 用于切向分母)
        self.input_cache = s.detach().clone() if self.training else None
        self.scale_factor = scale

        # 1. 幅相分离
        mag = torch.abs(s)           # |S|, shape 同 s
        phase = torch.angle(s)       # θ = angle(S)

        # 2. 强度筛选 (Softmax 仅作用于模长)
        scaled_mag = mag * scale
        mag_probs = torch.softmax(scaled_mag, dim=self.dim)   # p_k

        # 3. 全息重组: A = p · e^{iθ}
        out = torch.polar(mag_probs, phase)

        # 缓存前向中间量供反向使用
        if self.training:
            self.softmax_mag_cache = mag_probs    # p_k (Softmax 输出)
            self.phase_cache = phase              # θ_k (输入相位)

        return out

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float, **kwargs) -> torch.Tensor:
        """
        [修正版 v4.1] 精确 Wirtinger 反向传播 + 切向梯度平滑抑制。

        接收: G = grad_output = dL/dA*  (上游传来的复数误差向量)
        返回: dL/dS*                     (传给前层的复数误差向量)

        公式:
            dL/dS_j* = e_j · { scale · JVP_softmax(h)_j
                              + i · w(|S_j|) · p_j · t_j / (|S_j| + ε) }
        
        其中:
            e_j = e^{iθ_j}                                    (单位相位)
            h_k = Re(G_k · conj(e_k))                         (径向投影)
            t_j = Im(G_j · conj(e_j))                         (切向投影)
            JVP_softmax(h)_j = p_j · (h_j - Σ_k p_k · h_k)   (Softmax JVP)
            w(r) = r² / (r² + τ²)                             (平滑抑制, Fix #1)
        """
        # ---- 读取缓存 ----
        p = self.softmax_mag_cache       # p_k: Softmax 输出模长, 实数
        phase = self.phase_cache         # θ_k: 输入相位
        scale = self.scale_factor        # 缩放因子
        S = self.input_cache             # 原始复数输入 S

        # ---- 数值常量 ----
        eps = 1e-7                       # 除零保护 (纯安全网, 不影响正常数值)
        TAU_SQ = 1e-4                    # τ² = (0.01)², 切向抑制温度
        #
        # τ 的选择依据:
        #   · |S| > 10τ = 0.1 时: w > 0.99, 几乎无影响
        #   · |S| = τ = 0.01 时: w = 0.5, 切向力减半
        #   · |S| → 0 时: w → 0, 切向力完全关闭
        #
        # 这保证: 在 Q⊥K (|S|≈0) 的"相位模糊区"内不注入噪声,
        #          在 Q·K 有显著相干性时完全不干预。

        mag_S = torch.abs(S)             # |S_k|, 原始模长 (不加 eps, 供 w 计算)
        mag_S_safe = mag_S + eps         # 安全分母

        # ---- Step 1: 构造单位相位向量 e_k = e^{iθ_k} ----
        e = torch.polar(torch.ones_like(phase), phase)

        # ---- Step 2: 将上游梯度 G 分解为径向和切向分量 ----
        #
        # 在每个 k 位置，以 e_k 为径向基底:
        #   G_k = (径向投影) · e_k + (切向投影) · i·e_k
        #
        # h_k = Re(G_k · conj(e_k))  — 径向分量 (实数)
        # t_k = Im(G_k · conj(e_k))  — 切向分量 (实数)
        G_projected = grad_output * e.conj()       # G_k · e^{-iθ_k}
        h = torch.real(G_projected)                # 径向投影
        t = torch.imag(G_projected)                # 切向投影

        # ---- Step 3: 径向路径 — 标准 Softmax Jacobian-向量积 (JVP) ----
        #
        # JVP_j = p_j · (h_j - Σ_k p_k · h_k)
        #
        # 注意: 这里包含了所有交叉项 (k ≠ j)，
        #        原版只处理了对角项，导致梯度方向偏差。
        weighted_sum_h = torch.sum(p * h, dim=self.dim, keepdim=True)   # Σ_k p_k · h_k
        softmax_jvp = p * (h - weighted_sum_h)                          # JVP 完整结果

        # 径向梯度贡献: scale · JVP
        # 推导中 Wirtinger 的 1/2 因子与 2·Re(...) 的 2 精确抵消。
        radial_grad = scale * softmax_jvp          # 实数, shape 同 p

        # ---- Step 4: 切向路径 — 相位梯度 (含平滑抑制) ----
        #
        # [Fix #1] 引入平滑抑制函数:
        #     w(r) = r² / (r² + τ²)
        #
        # 原版: tangential = p · t / (|S| + 1e-4)
        #     |S|→0 时分母 ≈ 1e-4, 切向力可被放大 ~10000 倍
        #
        # 修正版: tangential = w · p · t / (|S| + eps)
        #     综合效果 = |S|² / (|S|² + τ²) · p · t / (|S| + eps)
        #             ≈ |S| / (|S|² + τ²) · p · t       (|S| 较小时)
        #     |S|→0 时整体 → 0, 数值安全
        mag_S_sq = mag_S.pow(2)
        tangential_weight = mag_S_sq / (mag_S_sq + TAU_SQ)

        tangential_grad = tangential_weight * p * t / mag_S_safe   # 实数

        # ---- Step 5: 合成复数梯度 ----
        #
        # dL/dS_j* = e_j · (radial + i · tangential)
        #
        # 径向分量沿 e_j 方向 (调节模长)
        # 切向分量沿 i·e_j 方向 (旋转相位)
        grad_input = e * (radial_grad + 1j * tangential_grad)

        # ---- 清理缓存 ----
        self.clear_cache()
        self.softmax_mag_cache = None
        self.phase_cache = None

        return grad_input


class HolographicAttention(ComplexLayer):
    """
    [Project Anla 核心组件] 全息共振注意力 (Holographic Resonance Attention)
    
    不同于 Transformer 的 "Query-Key Lookup"，
    这是一个 "Wave Interference & Transport" (波干涉与输运) 系统。
    
    原理:
    1. Interference: S = Q @ K^H (共轭干涉)
    2. Filtering: MagPhaseSoftmax (强度筛选，相位透传)
    3. Transport: O = A @ V (相干输运，自动补偿相位延迟)
    """

    def __init__(self, d_model, num_heads=8, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 1. 波函数投影 (Projection Layers)
        # Q, K, V, Output 都是全复数线性层
        self.w_q = ComplexLinear(d_model, d_model, bias=False)
        self.w_k = ComplexLinear(d_model, d_model, bias=False)
        self.w_v = ComplexLinear(d_model, d_model, bias=False)
        self.w_o = ComplexLinear(d_model, d_model, bias=False)

        # 2. 核心算子
        self.activation = MagPhaseSoftmax(dim=-1)

        # Cache for backward
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.attn_cache = None   # A matrix

    def _split_heads(self, x):
        """(Batch, Seq, Dim) -> (Batch, Heads, Seq, Head_Dim)"""
        new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        """(Batch, Heads, Seq, Head_Dim) -> (Batch, Seq, Dim)"""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.shape[:-2] + (self.d_model,)
        return x.view(*new_shape)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        前向传播: 投影 → 共轭干涉 → 幅相筛选 → 相干输运 → 输出投影

        Args:
            x: 输入张量, shape (Batch, Seq, Dim), dtype=complex
            mask: 可选的注意力遮罩 (暂未启用)

        Returns:
            输出张量, shape 同输入 (Batch, Seq, Dim)
        """
        # 1. Projections: 复数线性变换旋转并缩放波包
        q = self._split_heads(self.w_q(x))     # (B, H, Sq, Hd)
        k = self._split_heads(self.w_k(x))     # (B, H, Sk, Hd)
        v = self._split_heads(self.w_v(x))     # (B, H, Sk, Hd)

        # 缓存供反向使用
        if self.training:
            self.q_cache = q.detach().clone()
            self.k_cache = k.detach().clone()
            self.v_cache = v.detach().clone()

        # 2. Conjugate Interference (共轭干涉): S = Q @ K^H
        #    物理含义: 计算波的相干性。实部=强度匹配, 虚部=相对相位差。
        k_H = k.transpose(-1, -2).conj()       # K^H: (B, H, Hd, Sk)
        scores = torch.matmul(q, k_H)           # S:   (B, H, Sq, Sk)

        # Scale (归一化能量密度)
        scale = 1.0 / math.sqrt(self.head_dim)

        # Apply Mask (如有)
        if mask is not None:
            # 未来扩展: 在 magnitude 上施加遮罩
            pass

        # 3. Mag-Phase Activation: MagSoftmax * UnitPhase
        attn_weights = self.activation.forward(scores, scale=scale)

        if self.training:
            self.attn_cache = attn_weights.detach().clone()

        # 4. Coherent Transport (相干输运): O = A @ V
        #    V 中的波被 A 的相位旋转并按强度叠加
        attn_out = torch.matmul(attn_weights, v)   # (B, H, Sq, Hd)

        # 5. Output Projection
        out = self._combine_heads(attn_out)         # (B, Sq, Dim)
        out = self.w_o(out)

        return out

    def manual_backward(self, grad_output: torch.Tensor,
                        learning_rate: float,
                        weight_decay: float = 0.0) -> torch.Tensor:
        """
        全手动反向传播流程。

        梯度约定: 所有 grad 变量均为 Wirtinger 共轭导数 dL/d(·)*。

        反向路径:
            grad_output (dL/d_out*)
            → w_o backward  → grad_o      (dL/d(attn_combined)*)
            → split heads   → grad_attn_out (dL/dO*)
            → transport bwd → grad_A, grad_V
            → activation bwd→ grad_S
            → interference  → grad_Q, grad_K
            → w_q/k/v bwd  → grad_input (三路求和)

        推导验证 (transport 反向, O = A @ V):
            O 关于 A 全纯 → dL/dA* = dL/dO* @ V^{*T}
            O 关于 V 全纯 → dL/dV* = A^H @ dL/dO*

        推导验证 (interference 反向, S = Q @ K^H):
            S 关于 Q 全纯     → dL/dQ* = dL/dS* @ K
            S 关于 K 共轭线性 → dL/dK* = (dL/dS*)^H @ Q
        """
        # ---- 1. Output Linear Backward ----
        # grad_output: (Batch, Seq, Dim)
        grad_o = self.w_o.manual_backward(grad_output, learning_rate, weight_decay)

        # 恢复多头形状
        grad_attn_out = self._split_heads(grad_o)    # (B, H, Sq, Hd)

        # 读取前向缓存
        q = self.q_cache
        k = self.k_cache
        v = self.v_cache
        attn_weights = self.attn_cache               # A: (B, H, Sq, Sk)

        # ---- 2. Transport Backward: O = A @ V ----
        #
        # dL/dV* = A^H @ dL/dO*
        #   O_{ij} = Σ_k A_{ik} V_{kj}，O 关于 V 全纯
        #   ∂O*_{ij}/∂V*_{mn} = conj(A_{im}) · δ_{jn}
        #   → dL/dV*_{mn} = Σ_i G^O_{in} · conj(A_{im}) = [A^H @ G^O]_{mn}
        grad_v = torch.matmul(
            attn_weights.transpose(-1, -2).conj(),   # A^H: (B, H, Sk, Sq)
            grad_attn_out                             # G^O: (B, H, Sq, Hd)
        )                                             # → (B, H, Sk, Hd)

        # dL/dA* = dL/dO* @ V^{*T}
        #   O 关于 A 全纯
        #   ∂O*_{ij}/∂A*_{mn} = δ_{im} · conj(V_{nj})
        #   → dL/dA*_{mn} = Σ_j G^O_{mj} · conj(V_{nj}) = [G^O @ V^{*T}]_{mn}
        grad_attn_weights = torch.matmul(
            grad_attn_out,                            # G^O: (B, H, Sq, Hd)
            v.transpose(-1, -2).conj()                # V^{*T}: (B, H, Hd, Sk)
        )                                             # → (B, H, Sq, Sk)

        # ---- 3. Activation Backward: MagPhaseSoftmax ----
        grad_scores = self.activation.manual_backward(
            grad_attn_weights, learning_rate
        )                                             # → (B, H, Sq, Sk)

        # ---- 4. Interference Backward: S = Q @ K^H ----
        #
        # dL/dQ* = dL/dS* @ K
        #   S = Q @ K^H，S 关于 Q 全纯 → ∂S/∂Q* = 0
        #   ∂S*_{ij}/∂Q*_{mn} = δ_{im} · K_{jn}
        #   → dL/dQ*_{mn} = Σ_j G^S_{mj} · K_{jn} = [G^S @ K]_{mn}
        grad_q = torch.matmul(
            grad_scores,                              # G^S: (B, H, Sq, Sk)
            k                                         # K:   (B, H, Sk, Hd)
        )                                             # → (B, H, Sq, Hd)

        # dL/dK* = (dL/dS*)^H @ Q
        #   S 关于 K 共轭线性: S_{ij} = Σ_d Q_{id} · conj(K_{jd})
        #   ∂S_{ij}/∂K*_{mn} = δ_{jm} · Q_{in}
        #   → dL/dK*_{mn} = Σ_i conj(G^S_{im}) · Q_{in} = [(G^S)^H @ Q]_{mn}
        grad_k = torch.matmul(
            grad_scores.transpose(-1, -2).conj(),     # (G^S)^H: (B, H, Sk, Sq)
            q                                         # Q:       (B, H, Sq, Hd)
        )                                             # → (B, H, Sk, Hd)

        # ---- 5. Input Linear Backwards (Q, K, V) ----
        # 合并多头 → 通过各投影层反向 → 三路梯度求和
        grad_q_combined = self._combine_heads(grad_q)     # (B, Sq, Dim)
        grad_k_combined = self._combine_heads(grad_k)     # (B, Sq, Dim)
        grad_v_combined = self._combine_heads(grad_v)     # (B, Sq, Dim)

        dq = self.w_q.manual_backward(grad_q_combined, learning_rate, weight_decay)
        dk = self.w_k.manual_backward(grad_k_combined, learning_rate, weight_decay)
        dv = self.w_v.manual_backward(grad_v_combined, learning_rate, weight_decay)

        # 三路梯度叠加 (Q、K、V 共享同一输入 x)
        grad_input = dq + dk + dv

        # 清理缓存
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.attn_cache = None

        return grad_input
