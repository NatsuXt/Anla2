"""
Script: train_ring_transformer.py
Path: Anla/experiments/ring/train_ring_transformer.py
Description: 
    基于项目现有的 Transformer 架构 (AnlaManifoldModel) 训练哈密顿回路 (Hamiltonian Ring)。
    
    Reference: 
    - 架构源自: train_manifold_permuted.py
    - 任务目标: 验证模型能否在完全乱序的 ID 映射中重构出拓扑环。
    
    Key Features:
    1. 使用 ComplexTransformerBlock 替代简单的 Attention。
    2. 采用 manual_backward 物理动力学训练 (无 Autograd/Optimizer)。
    3. 可视化 View B 使用 "Hamiltonian Order" 着色，预期出现闭合环。
"""

import torch
import time
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ================= 路径与导入设置 =================
# 文件位置: Anla/experiments/ring/train_ring_transformer.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.positional import ComplexRotaryEmbedding

# ================= 可视化引擎 (适配环形拓扑) =================
class RingVisualizer:
    def __init__(self, base_dir):
        # 保存到 Logs/ring_transformer_vis
        self.log_dir = os.path.join(base_dir, "Logs", "ring_transformer_vis")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"--- Visualization Logs: {self.log_dir} ---")
        self.mse_history = []
        self.energy_history = []
        
    def update_stats(self, mse, energy):
        self.mse_history.append(mse)
        self.energy_history.append(energy)
        
    def plot_snapshot(self, epoch, model_embed, topo_order, acc, title_suffix=""):
        """
        topo_order: 节点在哈密顿环中的真实顺序 (0..N-1)。
        """
        weights = model_embed.weight.detach().cpu()
        # 拼接实部虚部 [vocab, 2*dim]
        weights_flat = torch.cat([weights.real, weights.imag], dim=1).numpy()
        
        # PCA 投影
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(weights_flat)
        
        # t-SNE 投影 (修复参数警告)
        coords_tsne = None
        # 仅在每 5 次绘图或最后阶段计算 t-SNE 以节省时间
        if epoch % 500 == 0 or "FINAL" in title_suffix or acc > 0.95:
            try:
                # [FIX] 移除 n_iter 参数，提高 perplexity 以适应环状结构
                tsne = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto')
                coords_tsne = tsne.fit_transform(weights_flat)
            except Exception as e:
                print(f"t-SNE Error: {e}")

        plt.figure(figsize=(24, 12))
        
        # Panel 1: 动力学曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.mse_history, label='MSE Loss')
        plt.yscale('log')
        plt.title(f'Dynamics (Acc: {acc:.2%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Panel 2: PCA - View A (Physical ID)
        # 颜色 = ID。如果 ID 是乱序的，这里应该是一团乱麻。
        colors_id = np.arange(len(weights))
        plt.subplot(2, 3, 2)
        plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=colors_id, cmap='turbo', s=30, alpha=0.8)
        plt.title('PCA View A: Token IDs (Expected Chaos)')
        plt.colorbar(label='Token ID')

        # Panel 3: PCA - View B (Topological Logic)
        # 颜色 = 环上的位置。如果成功，应显示彩虹环。
        plt.subplot(2, 3, 3)
        sc = plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=topo_order, cmap='hsv', s=50, alpha=0.9)
        plt.title('PCA View B: Hamiltonian Order (Hidden Ring)')
        plt.colorbar(sc, label='Ring Position')
        
        # 绘制连接线 (验证连续性)
        if acc > 0.8: # 准确率高时才画线，否则全是乱线
            sorted_idx = np.argsort(topo_order)
            sorted_xy = coords_pca[sorted_idx]
            plt.plot(sorted_xy[:, 0], sorted_xy[:, 1], c='gray', alpha=0.2, linewidth=0.5)
            # 闭环
            plt.plot([sorted_xy[-1, 0], sorted_xy[0, 0]], [sorted_xy[-1, 1], sorted_xy[0, 1]], c='gray', alpha=0.2)

        # Panel 5/6: t-SNE Views
        if coords_tsne is not None:
            plt.subplot(2, 3, 5)
            plt.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=colors_id, cmap='turbo', s=30, alpha=0.8)
            plt.title('t-SNE View A: Token IDs')
            
            plt.subplot(2, 3, 6)
            sc_tsne = plt.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=topo_order, cmap='hsv', s=50, alpha=0.9)
            plt.title('t-SNE View B: Hamiltonian Order')
            
            # t-SNE 连线
            if acc > 0.8:
                sorted_idx_tsne = np.argsort(topo_order)
                sorted_xy_tsne = coords_tsne[sorted_idx_tsne]
                plt.plot(sorted_xy_tsne[:, 0], sorted_xy_tsne[:, 1], c='gray', alpha=0.3, linewidth=0.8)
                plt.plot([sorted_xy_tsne[-1, 0], sorted_xy_tsne[0, 0]], [sorted_xy_tsne[-1, 1], sorted_xy_tsne[0, 1]], c='gray', alpha=0.3)
            
        save_path = os.path.join(self.log_dir, f"epoch_{epoch:05d}{title_suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# ================= 模型定义 (保持原架构一致性) =================
class AnlaManifoldModel(torch.nn.Module):
    def __init__(self, vocab_size, dim, layers=2, heads=4):
        super().__init__()
        # 1. 嵌入与位置编码
        self.embed = ComplexEmbedding(vocab_size, dim)
        self.pos_enc = ComplexRotaryEmbedding(dim, max_seq_len=2048)
        
        # 2. Transformer 堆叠
        self.blocks = torch.nn.ModuleList([
            ComplexTransformerBlock(dim, num_heads=heads) 
            for _ in range(layers)
        ])

    def forward(self, x):
        # x: [batch, seq_len]
        h = self.embed.forward(x)       # [batch, seq, dim]
        h = self.pos_enc.forward(h)     # Rotary Encoding
        for block in self.blocks:
            h = block.forward(h)
        return h

    def manual_backward(self, grad_output, lr, weight_decay):
        """
        全手动反向传播 (Path A)
        """
        grad = grad_output
        # 反向遍历 Transformer Block
        for block in reversed(self.blocks):
            grad = block.manual_backward(grad, lr, weight_decay)
        
        # 反向通过位置编码
        grad = self.pos_enc.manual_backward(grad)
        
        # 反向通过 Embedding (更新输入侧权重)
        self.embed.manual_backward(grad, lr, weight_decay)

# ================= 主训练逻辑 =================
def train_ring(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis = RingVisualizer(base_dir=_ANLA_ROOT)
    
    print(f"--- ANLA Experiment: Hamiltonian Ring Topology ---")
    print(f"--- Model: Transformer (Layers={args.layers}, Dim={args.dim}) ---")
    
    # 1. 初始化模型
    model = AnlaManifoldModel(args.vocab_size, args.dim, args.layers, args.heads).to(device)
    
    # 初始化缩放 (防止冷启动能量过大)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1: p.data.mul_(0.6)

    # 2. 构造哈密顿回路 (The Ring)
    # 这是一个 N -> N 的映射，但 ID 是乱的
    # mapping: x -> next_x
    # topo_order: x 在环中的位置 (用于 Ground Truth 可视化)
    print("Generating Hamiltonian Cycle...")
    path = np.random.permutation(args.vocab_size)
    mapping = np.zeros(args.vocab_size, dtype=int)
    for i in range(args.vocab_size - 1):
        mapping[path[i]] = path[i+1]
    mapping[path[-1]] = path[0] # 闭环
    
    topo_order = np.zeros(args.vocab_size, dtype=int)
    for step, node_idx in enumerate(path):
        topo_order[node_idx] = step
        
    # 转为 Tensor
    ring_map = torch.tensor(mapping, dtype=torch.long, device=device) # [vocab_size]
    
    start_time = time.time()
    converged = False 
    
    # 全量输入的批次索引 (用于计算准确率)
    all_indices = torch.arange(args.vocab_size, device=device)
    
    # 3. 训练循环
    for epoch in range(args.epochs + 1):
        # --- LR Schedule ---
        # 简单的线性预热 + 余弦衰减
        warmup_steps = 500
        if epoch < warmup_steps:
            current_lr = args.lr * (epoch + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (epoch - warmup_steps) / (args.epochs - warmup_steps)
            current_lr = args.lr * (0.5 * (1 + np.cos(np.pi * progress)))
            # 保持一个最小学习率，防止完全停止
            current_lr = max(current_lr, 1e-4)

        # --- Batch Data Generation ---
        # 随机采样 batch_size 个节点作为当前位置
        # Input shape: [Batch, Seq=1] (我们是在做一步预测)
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, 1)).to(device)
        
        # Target: 查表得到环上的下一个节点
        target_ids = ring_map[input_ids] # [Batch, 1]
        
        # --- Forward ---
        z_pred = model.forward(input_ids) # [Batch, 1, Dim]
        
        # 获取目标的波函数
        # 注意: target_ids 也是 [Batch, 1]
        z_target = model.embed.forward(target_ids) # [Batch, 1, Dim]
        
        # --- Physics: Force Calculation ---
        # 误差向量 = 预测位置 - 目标位置
        force_vector = z_pred - z_target # [Batch, 1, Dim]
        
        # 统计 MSE
        with torch.no_grad():
            mse_loss = 0.5 * (force_vector.abs().pow(2)).mean().item()
            avg_energy = model.embed.weight.abs().mean().item()
            vis.update_stats(mse_loss, avg_energy)

        # --- Path A: Network Update (Manual Backward) ---
        # 将误差传回网络
        model.manual_backward(force_vector, current_lr, args.weight_decay)
        
        # --- Path B: Manifold Evolution (Target Reaction) ---
        # Target 主动向 Prediction 移动 (双向纠缠)
        # 修正: reaction vector 应为 -force
        reaction_vector = -force_vector
        
        # 动态调整反作用力速率 (通常比 LR 小，以保持流形稳定)
        reaction_lr = current_lr * args.reaction_scale
        
        model.embed.manual_backward_explicit(
            grad=reaction_vector, # Force applied to target
            indices=target_ids,   # Which embeddings to move
            lr=reaction_lr,
            weight_decay=args.weight_decay
        )
        
        # --- Validation & Logging ---
        if epoch % 100 == 0:
            # 计算准确率 (Full Batch k-NN)
            model.eval() # 尽管没有 BatchNorm/Dropout，但也保持习惯
            with torch.no_grad():
                # 1. 预测所有节点的下一步
                # input: [vocab, 1]
                all_inputs_seq = all_indices.unsqueeze(1)
                all_z_pred = model.forward(all_inputs_seq).squeeze(1) # [vocab, dim]
                
                # 2. 计算与所有 Embedding 的距离
                # [vocab_pred, dim] vs [vocab_all, dim]
                # dist[i, j] = |pred[i] - emb[j]|^2
                # 为了省显存，我们分批计算或直接计算
                # 这里 vocab_size=200 很小，直接广播
                all_embeds = model.embed.weight # [vocab, dim]
                
                # (vocab, 1, dim) - (1, vocab, dim)
                dists = (all_z_pred.unsqueeze(1) - all_embeds.unsqueeze(0)).abs().pow(2).sum(dim=2)
                pred_ids = dists.argmin(dim=1) # [vocab]
                
                # 3. 比较
                targets = ring_map[all_indices]
                correct = (pred_ids == targets).sum().item()
                acc = correct / args.vocab_size
                
                print(f"Epoch {epoch:05d} | MSE: {mse_loss:.6f} | Acc: {acc:.2%} | LR: {current_lr:.5f}")
                
                # 绘图
                vis.plot_snapshot(epoch, model.embed, topo_order, acc)
                
                if acc == 1.0 and epoch > 1000:
                    print(f"\n[CONVERGED] Ring Topology Successfully Reconstructed!")
                    vis.plot_snapshot(epoch, model.embed, topo_order, acc, title_suffix="_FINAL")
                    converged = True
                    break
            model.train() # 切回训练状态

    if not converged:
        print("\n[Completed] Reached max epochs.")
        # Final check
        vis.plot_snapshot(args.epochs, model.embed, topo_order, acc, title_suffix="_FINAL")

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=200) 
    # [Config] 建议使用 32 或 64 维，以配合 Transformer 的多头注意力
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # [New] 反作用力缩放因子
    parser.add_argument("--reaction_scale", type=float, default=0.1)
    
    args = parser.parse_args()
    train_ring(args)
