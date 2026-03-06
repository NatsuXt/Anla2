"""
保存位置: Anla/experiments/ring/train_ring_masking.py

Anla Ring Inpainting — 流形修复训练脚本 (Manifold Repair)
=========================================================================

任务描述:
    在 64 节点的有序环 (0→1→2→...→63→0) 上生成长度 32 的子序列,
    随机遮蔽一个 span (1~5 tokens), 让模型通过双向上下文修复缺失段。
    
    这是一个 **拓扑完形填空** 任务:
    正确答案完全由环的结构和上下文决定,
    模型必须理解序列的拓扑连续性才能修复。

训练范式 (与原版的根本区别):
    原版: abs(logits) → CrossEntropy → polar(grad, phase) 贴回复数
          ↑ 这是传统 MLM 分类器范式, 输出端杀死相位, 梯度语义不一致
    
    本版: Transformer 输出复数状态 z_pred → 与目标原型 e_target 计算
          L_Elegant (对数模长差 + 单位相位差) → 得到复数力向量 → 直传 backward
          ↑ 这是流形修复范式, 相位是第一公民, 误差信号保持复数向量形态

判定函数 (L_Elegant):
    L = ( ln|z| - ln|ẑ| )²  +  | z/|z| - ẑ/|ẑ| |²
    
    第一项: 对数模长差 — 度量"能量尺度"的不匹配 (尺度不变, 防止大模长主导)
    第二项: 单位向量差 — 度量"相位方向"的不匹配 (纯几何, 与模长解耦)
    
    这个损失函数的关键优势:
    · 模长误差和相位误差在量纲上天然可比 (都是无量纲的)
    · 不存在"大模长压过相位"的问题
    · 梯度同时包含径向分量 (修正模长) 和切向分量 (修正相位)

力向量 (L_Elegant 对 z* 的 Wirtinger 梯度):
    令 r=|z|+ε, r̂=|ẑ|+ε, u=z/r, û=ẑ/r̂
    
    dL/dz* = [ ln(r/r̂) / r ] · u   +   [ u²·conj(û) - û ] / (2r)
              ╰───── 径向力 ─────╯       ╰──── 切向力 (扭矩) ────╯

评估方式:
    最近邻匹配 — 将 z_pred 与 embedding 表中所有 token 原型做距离比较,
    取最近的作为预测。不依赖投影头或 softmax。

v3 变更:
    [v3-1] Path B (target 侧更新) 使用 ρ(t) 软缩放:
        ρ(t) = clamp((L₀ - L(t)) / L₀, 0, 1)
        η_reaction = η · α · ρ(t)
        
        物理含义: ρ(t) 度量 Transformer 从上下文提取信息的能力。
        训练初期 ρ≈0, Path B 几乎不起作用 (prediction 不可靠);
        训练后期 ρ→1, Path B 以全力缩放运行 (prediction 可信)。

    [v3-2] 删除 Embedding RMS 稳态 (见 embedding.py v3),
           新增 embedding_rms 监控指标。
"""

import torch
import torch.nn as nn
import random
import sys
import os
import math

# -------------------------------------------------------------------------
# [Path Fix] 定位 Anla 包根目录与项目根目录
# 文件位置: Anla/experiments/ring/train_ring_masking.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock

import matplotlib.pyplot as plt

# ================= 加入以下两行全局配置 =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

# =====================================================================
#  全局配置
# =====================================================================
CONFIG = {
    # --- 任务参数 ---
    'vocab_size': 64,          # 环上的节点数
    'seq_len': 32,             # 序列长度
    'mask_token_id': 64,       # MASK token 的 ID (= vocab_size)
    'max_span_length': 5,      # 最大遮蔽跨度

    # --- 模型参数 ---
    'd_model': 64,             # 复数嵌入维度
    'num_heads': 4,            # 注意力头数

    # --- 训练参数 ---
    'batch_size': 16,
    'lr': 0.001,               # 学习率 (低于原版 0.005, 保护非线性)
    'weight_decay': 1e-4,      # 权重衰减 (能量耗散)
    'epochs': 5000,

    # [v3] Path B 反作用力参数
    # target 侧有效学习率 = lr * reaction_scale * ρ(t)
    'reaction_scale': 0.1,

    # --- 输出 ---
    'save_dir': 'checkpoints',
    'log_interval': 100,       # 每隔多少 epoch 打印一次日志
    'snapshot_interval': 500,  # 每隔多少 epoch 生成可视化快照
}


# =====================================================================
#  数据生成器 (与原版完全一致, 不做改动)
# =====================================================================
class RingSpanDataGenerator:
    """
    生成环形序列的 Span Masking 数据。
    
    · 从 [0, vocab_size) 的随机起点开始, 按 +1 mod vocab_size 生成有序序列
    · 随机选取一个 span (长度 1~max_span_length), 替换为 MASK token
    · target 中只有被 mask 的位置保留真实 token id, 其余为 -100
    """

    def __init__(self, vocab_size: int, seq_len: int, mask_id: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.mask_id = mask_id

    def generate_batch(self, batch_size: int):
        input_ids_list = []
        target_ids_list = []

        for _ in range(batch_size):
            # 1. 生成有序环序列
            start = random.randint(0, self.vocab_size - 1)
            seq = [(start + i) % self.vocab_size for i in range(self.seq_len)]
            seq_tensor = torch.tensor(seq, dtype=torch.long)

            # 2. 构造输入 (带 MASK) 和目标
            inp = seq_tensor.clone()
            tgt = torch.full_like(seq_tensor, -100)  # 默认忽略

            mask_len = random.randint(1, CONFIG['max_span_length'])
            mask_start = random.randint(0, self.seq_len - mask_len)

            inp[mask_start: mask_start + mask_len] = self.mask_id
            tgt[mask_start: mask_start + mask_len] = seq_tensor[mask_start: mask_start + mask_len]

            input_ids_list.append(inp)
            target_ids_list.append(tgt)

        return torch.stack(input_ids_list), torch.stack(target_ids_list)


# =====================================================================
#  模型定义 (移除了投影头, 回归流形修复范式)
# =====================================================================
class AnlaManifoldInpainter(nn.Module):
    """
    流形修复器: Embedding → Rotary → TransformerBlock → 复数状态输出
    
    与原版的关键区别:
    · 没有 ComplexLinear(d_model → vocab_size) 投影头
    · forward 直接返回 (Batch, Seq, d_model) 的复数张量
    · 预测通过最近邻匹配完成, 不通过 argmax(abs(logits))
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int):
        super().__init__()
        # vocab_size + 1: 额外一个位置留给 MASK token
        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=128)
        # Block 内部不使用 causal mask (双向注意力, 适合完形填空)
        self.block = ComplexTransformerBlock(d_model, num_heads=num_heads)
        # ★ 无投影头

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Seq) — 整数 token ids
        返回: (Batch, Seq, d_model) — 复数状态向量
        """
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)
        z_out = self.block.forward(z, mask=None)
        return z_out  # 直接返回复数状态, 不做任何降维/取模

    def manual_backward(self, force: torch.Tensor, lr: float, wd: float):
        """
        force: (Batch, Seq, d_model) — 复数力向量 (= dL/dz_out*)
               非 masked 位置应已置零。
        """
        # 逆序穿过各层
        grad = self.block.manual_backward(force, lr, wd)
        grad = self.rotary.manual_backward(grad)
        # [v3] Embedding 层不施加 weight decay:
        # L_Elegant 是尺度不变的 (度量 ln(r/r̂) 比值),
        # 无法感知全局模长缩放, 因此无力抵抗 weight decay 的无条件收缩,
        # 导致 EmbRMS 单调跌落至零。
        # Transformer 内部层的 weight decay 保留 (它们有足够的梯度对抗收缩)。
        self.embedding.manual_backward(grad, lr, weight_decay=0.0)


# =====================================================================
#  L_Elegant: 判定函数与力向量计算
# =====================================================================
def compute_elegant_loss_and_force(
    z_pred: torch.Tensor,       # (Batch, Seq, d_model), complex
    z_target: torch.Tensor,     # (Batch, Seq, d_model), complex
    valid_mask: torch.Tensor,   # (Batch, Seq), bool — True = masked position
) -> tuple:
    """
    计算 L_Elegant 及其 Wirtinger 梯度 (力向量)。
    
    L_Elegant = ( ln|z| - ln|ẑ| )²  +  | z/|z| - ẑ/|ẑ| |²
    
    力向量 = dL/dz* = [ ln(r/r̂)/r ]·u  +  [ u²·conj(û) - û ]/(2r)
    
    返回:
        loss_scalar  — 标量能量值 (仅用于监控)
        force        — (Batch, Seq, d_model) 复数力向量 (非 masked 位置为零)
    """
    eps = 1e-8

    # 模长和单位方向
    r = z_pred.abs() + eps           # (Batch, Seq, d_model), real
    r_hat = z_target.abs() + eps     # 同上
    u = z_pred / r                   # 近似单位复数
    u_hat = z_target / r_hat         # 同上

    # ----- 能量值 (标量, 仅监控) -----
    # 第一项: 对数模长差
    log_ratio = torch.log(r) - torch.log(r_hat)   # real
    loss_mag = log_ratio.pow(2)                    # real

    # 第二项: 单位向量差
    loss_phase = (u - u_hat).abs().pow(2)          # real

    # 逐元素损失 (Batch, Seq, d_model)
    loss_elem = loss_mag + loss_phase

    # ----- 力向量 (复数, 驱动反向传播) -----
    #
    # dL/dz* = 径向力 + 切向力
    #
    # 径向力: [ ln(r/r̂) / r ] · u
    #   含义: 当 r > r̂ 时指向内 (收缩), r < r̂ 时指向外 (膨胀)
    #   对数使得力与"比例"相关而非绝对值, 防止大模长主导
    #
    # 切向力: [ u²·conj(û) - û ] / (2r)
    #   含义: 在单位圆上旋转 u 使其靠近 û
    #   当 u ≈ û 时此项 → 0 (已对齐)
    #   推导详见文件头部注释
    #
    force_radial = (log_ratio / r) * u
    force_tangential = (u * u * u_hat.conj() - u_hat) / (2.0 * r)
    force = force_radial + force_tangential  # (Batch, Seq, d_model), complex

    # ----- 应用 mask: 非 masked 位置的力置零 -----
    # valid_mask: (Batch, Seq) → (Batch, Seq, 1)
    mask_3d = valid_mask.unsqueeze(-1).to(force.dtype)  # bool → complex
    force = force * mask_3d
    loss_elem = loss_elem * mask_3d.real  # 能量也只在 masked 位置统计

    # ----- 归一化: 按 masked token 总数取均值 -----
    num_valid = valid_mask.sum().float().clamp(min=1.0)
    force = force / num_valid   # 每个 token 的力独立于 batch 中 mask 数量
    loss_scalar = loss_elem.sum() / (num_valid * z_pred.shape[-1])  # 标量

    return loss_scalar.item(), force


# =====================================================================
#  最近邻评估
# =====================================================================
@torch.no_grad()
def evaluate_nearest_neighbor(
    z_pred: torch.Tensor,       # (Batch, Seq, d_model)
    target_ids: torch.Tensor,   # (Batch, Seq), -100 = 忽略
    all_embeds: torch.Tensor,   # (vocab_size, d_model), 所有 token 的 embedding
) -> tuple:
    """
    对 masked 位置用最近邻匹配进行预测评估。
    
    返回:
        accuracy   — 正确率 (float)
        n_correct  — 正确数
        n_total    — masked 位置总数
    """
    vocab_size = all_embeds.shape[0]
    valid_mask = (target_ids != -100)  # (Batch, Seq)

    if not valid_mask.any():
        return 0.0, 0, 0

    # 提取 masked 位置的预测向量
    z_masked = z_pred[valid_mask]          # (N_masked, d_model)
    true_ids = target_ids[valid_mask]      # (N_masked,)

    # 计算与所有 token embedding 的距离
    # z_masked: (N, 1, d_model)  -  all_embeds: (1, V, d_model)
    dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(dim=-1)
    # dists: (N_masked, vocab_size)

    pred_ids = dists.argmin(dim=-1)  # (N_masked,)

    n_correct = (pred_ids == true_ids).sum().item()
    n_total = true_ids.shape[0]
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    return accuracy, n_correct, n_total


# =====================================================================
#  Checkpoint 工具
# =====================================================================
def save_checkpoint(model, config, filename):
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, path)


# =====================================================================
#  主训练循环
# =====================================================================
def train_ring_inpainting():
    print("=" * 70)
    print(" Anla Ring Inpainting — Manifold Repair (L_Elegant)")
    print("=" * 70)
    print(f"Configuration: {CONFIG}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ----- 初始化 -----
    generator = RingSpanDataGenerator(
        CONFIG['vocab_size'], CONFIG['seq_len'], CONFIG['mask_token_id'])

    model = AnlaManifoldInpainter(
        CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['num_heads']
    ).to(device)

    # ----- 可视化引擎初始化 -----
    # 导入可视化模块 (如果不存在则跳过, 不影响训练)
    vis = None
    try:
        from Anla.visualization.visualize_ring_masking import RingMaskingVisualizer
        vis = RingMaskingVisualizer(base_dir=_ANLA_ROOT, vocab_size=CONFIG['vocab_size'])
        print("[Vis] 可视化引擎已启用")
    except ImportError:
        print("[Vis] 可视化模块未找到, 将跳过可视化 (不影响训练)")

    best_acc = 0.0
    best_loss = float('inf')

    # ----- [v3] ρ(t) 学习进度追踪 -----
    #
    # ρ(t) = clamp((L₀ - L(t)) / L₀, 0, 1)
    #
    # L₀: 初始 loss (前 N 步的 EMA 均值)
    # L(t): 当前 smoothed loss
    #
    # 用途: 控制 Path B (target 侧更新) 的有效学习率
    #   η_reaction = η · reaction_scale · ρ(t)
    #
    # 物理含义: ρ(t) 度量 Transformer 从上下文提取信息的能力
    #   ρ ≈ 0 → 模型未学到东西, prediction 不可靠, Path B 几乎关闭
    #   ρ → 1 → loss 趋近零, prediction 高度可信, Path B 全力运行
    rho = 0.0                # 当前学习进度
    initial_loss = None      # L₀, 在第一步设定
    smoothed_loss = None     # L(t), EMA 平滑
    rho_ema_beta = 0.95      # smoothed loss 的 EMA 系数

    # ----- 训练循环 -----
    for epoch in range(CONFIG['epochs']):
        model.train()

        # 1. 生成数据
        input_ids, target_ids = generator.generate_batch(CONFIG['batch_size'])
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 2. 前向传播 → 复数状态输出
        z_pred = model.forward(input_ids)  # (Batch, Seq, d_model), complex

        # 3. 获取目标原型 (直接从 embedding weight 索引, 不走 forward!)
        #
        # ★ 关键实现细节:
        #    不能使用 model.embedding.forward(target_ids), 因为那会覆盖
        #    embedding 层的 input_cache, 导致后续 manual_backward 更新
        #    错误的 token。
        #
        #    正确做法: 直接用 weight 矩阵索引, 并 detach。
        #
        safe_target_ids = target_ids.clone()
        safe_target_ids[target_ids == -100] = 0  # 用 0 占位, 后续 mask 会屏蔽
        z_target = model.embedding.weight.data[safe_target_ids].detach()
        # z_target: (Batch, Seq, d_model), complex

        # 4. 计算 L_Elegant 力向量
        valid_mask = (target_ids != -100)  # (Batch, Seq), bool
        loss_val, force = compute_elegant_loss_and_force(z_pred, z_target, valid_mask)

        # 5. 手动反向传播 (复数力向量直接传入, 无缝合)
        model.manual_backward(force, CONFIG['lr'], CONFIG['weight_decay'])

        # 6. [v3] 更新 ρ(t) 学习进度
        #
        # 首步: 记录 L₀ (初始 loss)
        # 后续: EMA 平滑当前 loss, 计算 ρ = clamp((L₀ - L_smoothed) / L₀, 0, 1)
        if initial_loss is None:
            initial_loss = loss_val
            smoothed_loss = loss_val
        else:
            smoothed_loss = rho_ema_beta * smoothed_loss + (1.0 - rho_ema_beta) * loss_val

        if initial_loss > 1e-12:
            rho = max(0.0, min(1.0, (initial_loss - smoothed_loss) / initial_loss))
        else:
            rho = 0.0

        # 7. [v3] Path B: Target 侧 embedding 更新 (ρ(t) 软缩放)
        #
        # 有效学习率 = lr * reaction_scale * ρ(t)
        #
        # 训练初期 ρ≈0: Path B 几乎不施力 (prediction 方向不可靠)
        # 训练后期 ρ→1: Path B 以 reaction_scale 的全力运行
        #
        # 力方向 = -force (反作用力: 让 target 向 prediction 微弱移动)
        reaction_lr = CONFIG['lr'] * CONFIG['reaction_scale'] * rho
        if reaction_lr > 1e-12 and valid_mask.any():
            valid_target_ids = target_ids[valid_mask]
            valid_reaction = -force[valid_mask]
            model.embedding.manual_backward_explicit(
                grad=valid_reaction,
                indices=valid_target_ids,
                lr=reaction_lr,
                weight_decay=0.0,  # [v3] embedding 不施加 weight decay
            )

        # ----- 日志与评估 -----
        if epoch % CONFIG['log_interval'] == 0:
            model.eval()
            with torch.no_grad():
                # 重新前向 (eval 模式不会更新缓存)
                z_eval = model.forward(input_ids)

                # 获取 embedding 原型 (排除 MASK token)
                all_embeds = model.embedding.weight.data[:CONFIG['vocab_size']]

                # 最近邻评估
                acc, n_ok, n_tot = evaluate_nearest_neighbor(
                    z_eval, target_ids, all_embeds)

            # 记录到可视化引擎
            if vis is not None:
                vis.record(epoch, loss_val, acc)

            # 打印日志 (含 ρ 学习进度)
            print(f"Epoch {epoch:04d} | L_Elegant: {loss_val:.6f} | "
                  f"Acc: {acc:.2%} ({n_ok}/{n_tot}) | ρ: {rho:.3f}")

            # Debug: 显示一个样本的修复情况
            sample_mask = (target_ids[0] != -100)
            if sample_mask.any():
                idx = torch.where(sample_mask)[0][0].item()
                start = max(0, idx - 2)
                end = min(CONFIG['seq_len'], idx + 3)

                context = input_ids[0, start:end].tolist()
                disp = ['[M]' if x == CONFIG['mask_token_id'] else str(x)
                        for x in context]

                true_val = target_ids[0, idx].item()

                # 最近邻预测
                z_at_idx = z_eval[0, idx]  # (d_model,)
                dists = (z_at_idx.unsqueeze(0) - all_embeds).abs().pow(2).sum(dim=-1)
                pred_val = dists.argmin().item()

                status = "✓" if true_val == pred_val else "✗"
                print(f"   [{status}] ...{' '.join(disp)}... → "
                      f"True: {true_val} | Pred: {pred_val}")

            # 保存最佳模型
            if acc > best_acc or (acc == best_acc and loss_val < best_loss):
                best_acc = acc
                best_loss = loss_val
                save_checkpoint(model, CONFIG, 'best_ring_inpainter.pth')

            # --- 诊断指标 (可选, 帮助判断系统健康度) ---
            if epoch % (CONFIG['log_interval'] * 5) == 0:
                emb_w = model.embedding.weight.data[:CONFIG['vocab_size']]
                emb_mag = emb_w.abs().mean().item()
                emb_phase_std = torch.angle(emb_w).std().item()

                # [v3] Embedding RMS (模长稳定性监控, 替代已删除的 RMS 稳态)
                emb_rms = torch.sqrt(emb_w.abs().pow(2).mean()).item()

                # PhaseTwist 参数监控
                act = model.block.act  # TransformerBlock 内的 PhaseTwist
                gamma_mean = act.gamma.data.abs().mean().item()
                beta_mean = act.beta.data.abs().mean().item()

                print(f"   [Diag] Emb|w|: {emb_mag:.4f} | "
                      f"EmbRMS: {emb_rms:.4f} | "
                      f"Phase σ: {emb_phase_std:.4f} | "
                      f"|γ|: {gamma_mean:.6f} | |β|: {beta_mean:.6f}")

            # --- 可视化快照 (定期生成) ---
            if vis is not None and epoch % CONFIG['snapshot_interval'] == 0:
                vis.snapshot(epoch, model.embedding, acc)

        model.train()

    # ----- 训练结束 -----
    save_checkpoint(model, CONFIG, 'final_ring_inpainter.pth')
    try:
        from Anla.analysis.topology_audit_ring import run_topology_audit_from_embedding
        run_topology_audit_from_embedding(
            model.embedding.weight.data[:CONFIG["vocab_size"]],
            output_dir=os.path.join(_ANLA_ROOT, "Logs", "ring_masking_vis"),
            initial_k=6,
            persistence_ratio=0.10,
            maxdim=2,
        )
    except Exception as exc:
        print(f"[TDA] topology audit skipped: {exc}")

    print()
    print("=" * 70)
    print(f" Training Complete. Best Accuracy: {best_acc:.2%}")
    print(f" Final ρ: {rho:.3f}")
    print("=" * 70)

    # ----- 最终可视化分析 -----
    if vis is not None:
        vis.final_analysis(model.embedding)

    # ----- 简要结果判定 -----
    if best_acc > 0.9:
        print("[SUCCESS] 流形修复有效: 模型成功学会了环拓扑的 span 填空。")
    elif best_acc > 0.3:
        print("[PARTIAL] 训练信号有效但未充分收敛, 建议增加 epoch 或调整 lr。")
    else:
        print("[INVESTIGATE] 准确率过低, 需要检查:")
        print("  1. 能量 (loss) 是否在下降")
        print("  2. Embedding RMS 是否在合理范围 (0.5~3.0)")
        print("  3. Gamma 是否归零 (线性坍缩征兆)")


# =====================================================================
#  入口
# =====================================================================
if __name__ == "__main__":
    train_ring_inpainting()
