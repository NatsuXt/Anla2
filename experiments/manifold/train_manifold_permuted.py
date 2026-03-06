import torch
import time
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------------------------------
# [Path Fix] 定位 Anla 包根目录与项目根目录
# 文件位置: Anla/experiments/manifold/train_manifold_permuted.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.positional import ComplexRotaryEmbedding

# ==========================================
#  Visualization Engine (Dual View)
# ==========================================
class ManifoldVisualizer:
    def __init__(self, base_dir):
        self.log_dir = os.path.join(base_dir, "Logs", "manifold_vis_permuted")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"--- Logs: {self.log_dir} ---")
        self.mse_history = []
        self.energy_history = []
        
    def update_stats(self, mse, energy):
        self.mse_history.append(mse)
        self.energy_history.append(energy)
        
    def plot_snapshot(self, epoch, model_embed, perm_order, title_suffix=""):
        """
        perm_order: 真实的逻辑顺序链条 (Logic Chain)
        我们不仅看 ID 分布，还要看 Logic 分布
        """
        weights = model_embed.weight.detach().cpu()
        weights_flat = torch.cat([weights.real, weights.imag], dim=1).numpy()
        
        # PCA
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(weights_flat)
        
        # t-SNE (仅在后期开启)
        coords_tsne = None
        if epoch > 1000 or "FINAL" in title_suffix:
            try:
                tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
                coords_tsne = tsne.fit_transform(weights_flat)
            except: pass

        plt.figure(figsize=(24, 12)) # 更大的画布
        
        # 1. 动力学
        plt.subplot(2, 3, 1)
        plt.plot(self.mse_history, label='MSE')
        plt.plot(self.energy_history, label='Energy')
        plt.yscale('log')
        plt.title('System Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ------------------------------------------------
        # 视角 A: 物理 ID 视图 (模拟真实观察)
        # 颜色由 Token ID 决定。如果任务是乱序的，这里应该看起来像乱麻/云团
        # ------------------------------------------------
        colors_id = np.arange(len(weights))
        
        plt.subplot(2, 3, 2)
        plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=colors_id, cmap='turbo', s=30, alpha=0.8)
        plt.title('View A: Physical IDs (Expected Chaos)')
        plt.colorbar(label='Token ID')

        # ------------------------------------------------
        # 视角 B: 逻辑链视图 (上帝视角)
        # 颜色由 "它在链条中的位置" 决定。
        # 如果模型学会了乱序规律，这里应该显现出 "彩虹/圆环"！
        # ------------------------------------------------
        # 计算每个点在逻辑链中的 rank
        # perm_order 是一个链条 [a -> b -> c ...]
        # 我们需要反向映射：Token X 是链条中的第几个？
        
        # 构造逻辑颜色映射
        logic_colors = np.zeros(len(weights))
        # 假设 perm_order 是 [0, 5, 92...] 这种链式访问顺序
        # 这里的 perm_order 实际上是 "Next Map". 
        # 为了可视化"链条"，我们需要追踪轨迹： 0 -> next -> next ...
        # 这在代码外部计算好传入更佳。这里简化处理，假设 perm_order 本身定义了另一种拓扑
        
        # [Visual Logic Correction]
        # 如果我们想看"彩虹"，我们需要按照"真实的语义顺序"来着色。
        # 在 random permutation 任务中，我们可以追踪一条长路径。
        # 简单起见，我们直接使用传入的 topological_rank
        plt.subplot(2, 3, 3)
        plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=perm_order, cmap='turbo', s=30, alpha=0.8)
        plt.title('View B: Logical Flow (Hidden Order)')
        plt.colorbar(label='Logical Rank')
        
        # t-SNE Views (同上)
        if coords_tsne is not None:
            plt.subplot(2, 3, 5)
            plt.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=colors_id, cmap='turbo', s=20, alpha=0.8)
            plt.title('t-SNE (Physical ID)')
            
            plt.subplot(2, 3, 6)
            plt.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=perm_order, cmap='turbo', s=20, alpha=0.8)
            plt.title('t-SNE (Logical Flow)')
            
        save_path = os.path.join(self.log_dir, f"epoch_{epoch:05d}{title_suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# ... (Model 类保持不变) ...
class AnlaManifoldModel(torch.nn.Module):
    def __init__(self, vocab_size, dim, layers=2, heads=4):
        super().__init__()
        self.embed = ComplexEmbedding(vocab_size, dim)
        self.pos_enc = ComplexRotaryEmbedding(dim, max_seq_len=2048)
        self.blocks = torch.nn.ModuleList([
            ComplexTransformerBlock(dim, num_heads=heads) 
            for _ in range(layers)
        ])

    def forward(self, x):
        h = self.embed.forward(x)
        h = self.pos_enc.forward(h)
        for block in self.blocks:
            h = block.forward(h)
        return h

    def manual_backward(self, grad_output, lr, weight_decay):
        grad = grad_output
        for block in reversed(self.blocks):
            grad = block.manual_backward(grad, lr, weight_decay)
        grad = self.pos_enc.manual_backward(grad)
        self.embed.manual_backward(grad, lr, weight_decay)

def train_manifold(args):
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    vis = ManifoldVisualizer(base_dir=_ANLA_ROOT)
    
    print(f"--- ANLA AGI: Realistic Language Simulation ---")
    print(f"--- Task: Deterministic Permutation (Chaos -> Order) ---")
    
    model = AnlaManifoldModel(args.vocab_size, args.dim, args.layers, args.heads).to(device)
    
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1: p.data.mul_(0.6)

    # =====================================================
    # [CORE CHANGE] 构建"乱序但确定"的语义宇宙
    # =====================================================
    # 1. 生成一个固定的乱序表 (The Laws of this Universe)
    # 这是一个随即排列，比如 [5, 92, 1, 30...]
    # 含义：ID 0 的下一个是 5，ID 1 的下一个是 92...
    # 这完全打破了 +1 的规律，模拟真实语言的任意性
    perm_map = torch.randperm(args.vocab_size, device=device)
    
    # 2. 为了可视化"逻辑链"，我们需要解算出链条的顺序
    # 我们从 0 开始追踪： 0 -> perm[0] -> perm[perm[0]] ...
    # 这构成了一条长链（或者几个环）。我们按这个访问顺序给点打上"逻辑排名"
    # 这样我们在 View B 中就能看到彩虹了！
    logical_ranks = torch.zeros(args.vocab_size, device=device)
    curr = 0
    visited = set()
    rank = 0
    # 简单起见，我们只追踪主环（Main Loop）
    for _ in range(args.vocab_size):
        if curr in visited: break
        visited.add(curr)
        logical_ranks[curr] = rank
        curr = perm_map[curr].item()
        rank += 1
    
    # 归一化 rank 用于绘图颜色
    logical_ranks_cpu = logical_ranks.cpu().numpy()

    print(f"Permutation Map Generated. Semantic Logic defined.")
    
    start_time = time.time()
    converged = False 
    
    for epoch in range(args.epochs):
        # LR Schedule
        warmup_steps = 500
        if epoch < warmup_steps:
            current_lr = args.lr * (epoch + 1) / warmup_steps
        else:
            progress = (epoch - warmup_steps) / (args.epochs - warmup_steps)
            current_lr = args.lr * (1.0 - progress * 0.9)

        # =================================================
        # 数据生成：完全随机采样，但遵循 perm_map 规律
        # =================================================
        # 1. 随机生成输入
        input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to(device)
        
        # 2. 目标必须查表！不能用 +1！
        # 使用 embedding 风格的查找将 input_ids 映射到 target_ids
        target_ids = perm_map[input_ids] # <--- 核心差异：查乱序表
        
        # === Forward ===
        z_pred = model.forward(input_ids)
        z_target = model.embed.forward(target_ids)
        
        # === Physics ===
        force_vector = z_pred - z_target
        
        with torch.no_grad():
            mse_loss = 0.5 * (force_vector.abs().pow(2)).mean().item()
            avg_energy = model.embed.weight.abs().mean().item()
            vis.update_stats(mse_loss, avg_energy)

        # === Bidi-Dynamics ===
        model.manual_backward(force_vector, current_lr, args.weight_decay)
        reaction_vector = -force_vector
        model.embed.manual_backward_explicit(
            grad=reaction_vector,
            indices=target_ids,
            lr=current_lr,
            weight_decay=args.weight_decay
        )
        
        # === Logging ===
        plot_interval = 100 if epoch < 1000 else 500
        if epoch % 50 == 0:
            print(f"Epoch {epoch:05d} | MSE: {mse_loss:.6f} | Energy: {avg_energy:.4f}")
        
        if epoch % plot_interval == 0:
            # 传入 logical_ranks 以绘制 View B
            vis.plot_snapshot(epoch, model.embed, logical_ranks_cpu)
            
            if mse_loss < 0.0005 and epoch > 1000:
                print("\n[Converged] Manifold stabilized.")
                vis.plot_snapshot(epoch, model.embed, logical_ranks_cpu, title_suffix="_FINAL")
                converged = True
                break
    
    if not converged:
        print("\n[Completed] Reached max epochs.")
        vis.plot_snapshot(args.epochs, model.embed, logical_ranks_cpu, title_suffix="_FINAL")

    print(f"Total Time: {time.time() - start_time:.2f}s")
    
    # =================================================
    # 科学验证：全量乱序映射测试
    # =================================================
    print("\n--- Full Validation: Permutation Logic ---")
    model.eval()
    
    all_vocab_emb = model.embed.weight.detach()
    total_correct = 0
    total_samples = 0
    
    # 我们遍历整个词表，看模型是否学会了 perm_map
    # 构造 batch
    batch_size_val = 100
    all_indices = torch.arange(args.vocab_size, device=device)
    
    with torch.no_grad():
        for i in range(0, args.vocab_size, batch_size_val):
            # Input IDs
            curr_input = all_indices[i : i+batch_size_val] # (B)
            # 扩展为 (B, 1) 或 (B, Seq) 以适配模型
            curr_input_seq = curr_input.unsqueeze(1).repeat(1, args.seq_len)
            
            # Ground Truth (查表)
            curr_target = perm_map[curr_input] # (B)
            
            # Forward
            z_out = model.forward(curr_input_seq) # (B, Seq, Dim)
            
            # Check last token prediction
            last_pred_vecs = z_out[:, -1, :] # (B, Dim)
            
            # Batch k-NN
            # (B, 1, Dim) - (1, Vocab, Dim) -> (B, Vocab, Dim)
            dists = (last_pred_vecs.unsqueeze(1) - all_vocab_emb.unsqueeze(0)).abs().pow(2).sum(dim=2)
            pred_ids = dists.argmin(dim=1) # (B)
            
            correct = (pred_ids == curr_target).sum().item()
            total_correct += correct
            total_samples += len(curr_input)
            
    acc = total_correct / total_samples * 100
    print(f"Global Chaos Accuracy: {acc:.2f}% ({total_correct}/{total_samples})")
    
    if acc > 99.0:
        print("[PERFECT] Anla successfully extracted Order from Chaos.")
    else:
        print("[FAIL] Dynamics failed to capture the permutation logic.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=200) 
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    args = parser.parse_args()
    train_manifold(args)
