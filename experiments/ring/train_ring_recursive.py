"""
Script: train_ring_recursive.py
Path: Anla/experiments/ring/train_ring_recursive.py
Description: 
    [Fix Version]
    修复了 activation.py 中缓存被覆盖导致的 NoneType 错误。
    策略：Step-by-Step Forward-Backward Interleaving.
    即：Forward(t) -> Backward(t) -> Forward(t+1) ...
    这样确保每次 Backward 时，层内的 cache 正好对应刚刚执行的 Forward。
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

# ================= 导入设置 =================
# 文件位置: Anla/experiments/ring/train_ring_recursive.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.positional import ComplexRotaryEmbedding

# ================= 可视化引擎 =================
class RingVisualizer:
    def __init__(self, base_dir, k_steps):
        self.log_dir = os.path.join(base_dir, "Logs", f"recursive_k{k_steps}_vis")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"--- Logs: {self.log_dir} ---")
        self.loss_history = []
        
    def plot(self, epoch, model, topo_order, acc):
        weights = model.embed.weight.detach().cpu()
        flat = torch.cat([weights.real, weights.imag], dim=1).numpy()
        
        pca = PCA(n_components=2)
        xy_pca = pca.fit_transform(flat)
        
        plt.figure(figsize=(16, 8))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.title(f'Recursive Loss (Acc: {acc:.1%})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        sc = plt.scatter(xy_pca[:, 0], xy_pca[:, 1], c=topo_order, cmap='hsv', s=60, alpha=0.8)
        plt.colorbar(sc, label='Hamiltonian Order')
        plt.title('PCA: Topological Structure')
        
        sorted_idx = np.argsort(topo_order)
        sorted_xy = xy_pca[sorted_idx]
        plt.plot(sorted_xy[:, 0], sorted_xy[:, 1], c='gray', alpha=0.3)
        plt.plot([sorted_xy[-1, 0], sorted_xy[0, 0]], [sorted_xy[-1, 1], sorted_xy[0, 1]], c='gray', alpha=0.3)
        
        if epoch % 200 == 0 or acc > 0.9:
            try:
                tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
                xy_tsne = tsne.fit_transform(flat)
                plt.subplot(1, 3, 3)
                sc_t = plt.scatter(xy_tsne[:, 0], xy_tsne[:, 1], c=topo_order, cmap='hsv', s=60, alpha=0.8)
                plt.title('t-SNE: Topological Structure')
                sorted_idx_t = np.argsort(topo_order)
                sorted_xy_t = xy_tsne[sorted_idx_t]
                plt.plot(sorted_xy_t[:, 0], sorted_xy_t[:, 1], c='gray', alpha=0.3)
                plt.plot([sorted_xy_t[-1, 0], sorted_xy_t[0, 0]], [sorted_xy_t[-1, 1], sorted_xy_t[0, 1]], c='gray', alpha=0.3)
            except:
                pass

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"epoch_{epoch:05d}.png"))
        plt.close()

# ================= 模型定义 =================
class AnlaModel(torch.nn.Module):
    def __init__(self, vocab_size, dim, layers=2, heads=4):
        super().__init__()
        self.embed = ComplexEmbedding(vocab_size, dim)
        self.pos_enc = ComplexRotaryEmbedding(dim)
        self.blocks = torch.nn.ModuleList([
            ComplexTransformerBlock(dim, num_heads=heads) for _ in range(layers)
        ])

    def forward_step(self, z_input):
        # z_input: [batch, 1, dim]
        h = self.pos_enc.forward(z_input)
        for block in self.blocks:
            h = block.forward(h)
        return h 

    def manual_backward(self, grad_output, lr, weight_decay):
        grad = grad_output
        for block in reversed(self.blocks):
            grad = block.manual_backward(grad, lr, weight_decay)
        grad = self.pos_enc.manual_backward(grad)
        return grad

# ================= 训练逻辑 =================
def train_recursive(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- ANLA Recursive Training (Fixed V2, K={args.k_steps}) ---")
    
    # 1. Setup
    model = AnlaModel(args.vocab_size, args.dim, args.layers, args.heads).to(device)
    vis = RingVisualizer(_ANLA_ROOT, args.k_steps)
    
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1: p.data.mul_(0.6)
            
    # 2. Data
    path = np.random.permutation(args.vocab_size)
    mapping = np.zeros(args.vocab_size, dtype=int)
    for i in range(args.vocab_size - 1):
        mapping[path[i]] = path[i+1]
    mapping[path[-1]] = path[0]
    ring_map = torch.tensor(mapping, dtype=torch.long, device=device)
    
    # Pre-calc targets
    targets_table = torch.zeros((args.vocab_size, args.k_steps), dtype=torch.long, device=device)
    curr = torch.arange(args.vocab_size, device=device)
    for k in range(args.k_steps):
        curr = ring_map[curr]
        targets_table[:, k] = curr
        
    topo_order = np.zeros(args.vocab_size, dtype=int)
    for step, node_idx in enumerate(path):
        topo_order[node_idx] = step

    start_time = time.time()
    
    # 3. Training Loop
    for epoch in range(args.epochs + 1):
        # LR Schedule
        lr = args.lr
        if epoch > 1000: lr *= 0.5
        if epoch > 3000: lr *= 0.2
        
        # Batch sampling
        start_ids = torch.randint(0, args.vocab_size, (args.batch_size, 1)).to(device)
        trajectory_ids = targets_table[start_ids.squeeze(1)] # [Batch, K]
        
        # Initial State
        current_z = model.embed.forward(start_ids) # [Batch, 1, Dim]
        
        total_loss_val = 0
        
        # --- Recursive Step-by-Step (Interleaved) ---
        # 核心修复: 在进行下一步 Forward 之前，必须完成这一步的 Backward
        # 这样 input_cache 才是对应的。
        
        for k in range(args.k_steps):
            # 1. Forward Step k
            next_z = model.forward_step(current_z) # Cache updated here
            
            # 2. Calculate Error immediately
            target_ids_k = trajectory_ids[:, k]
            z_target_k = model.embed.forward(target_ids_k.unsqueeze(1))
            
            # Error = Pred - Target
            force = next_z - z_target_k
            
            loss_k = 0.5 * force.abs().pow(2).mean().item()
            total_loss_val += loss_k
            
            # 3. Backward Step k immediately (consuming cache)
            # Scale gradient by 1/K
            model.manual_backward(force * (1.0/args.k_steps), lr, args.weight_decay)
            
            # 4. Path B Update
            reaction_force = -force
            model.embed.manual_backward_explicit(
                grad=reaction_force * args.reaction_scale * (1.0/args.k_steps),
                indices=target_ids_k,
                lr=lr * 0.1
            )
            
            # 5. Prepare for next step
            # We must use DETACHED prediction for next step forward to prevent 
            # PyTorch from trying to build a graph (which we don't use) 
            # and to save memory. 
            # In manual_backward mode, we only need the value.
            current_z = next_z.detach() 
            
        vis.loss_history.append(total_loss_val)
        
        # --- Eval ---
        if epoch % 100 == 0:
            model.eval()
            # Do a single step eval for metric
            all_ids = torch.arange(args.vocab_size, device=device).unsqueeze(1)
            # 注意: eval mode 下 forward 不更新 cache，所以是安全的
            z0 = model.embed.forward(all_ids)
            z1_pred = model.forward_step(z0).squeeze(1)
            
            all_w = model.embed.weight
            dists = (z1_pred.unsqueeze(1) - all_w.unsqueeze(0)).abs().pow(2).sum(dim=2)
            preds = dists.argmin(dim=1)
            targets = ring_map
            acc = (preds == targets).float().mean().item()
            
            print(f"Epoch {epoch:04d} | Loss: {total_loss_val:.4f} | Acc: {acc:.2%} | K={args.k_steps}")
            vis.plot(epoch, model, topo_order, acc)
            
            if acc >= 0.995 and epoch > 500:
                print("Converged!")
                break
            model.train()

    print(f"Total Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reaction_scale", type=float, default=0.05)
    parser.add_argument("--k_steps", type=int, default=5) 
    
    args = parser.parse_args()
    train_recursive(args)
