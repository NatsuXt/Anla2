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
# 文件位置: Anla/experiments/manifold/train_manifold.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))      # .../Anla/experiments/manifold
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))  # .../Anla
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))    # .../

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.positional import ComplexRotaryEmbedding

# ==========================================
#  Visualization Engine (Scientific Plotting)
# ==========================================
class ManifoldVisualizer:
    def __init__(self, base_dir):
        self.log_dir = os.path.join(base_dir, "Logs", "manifold_vis")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"--- Visualization Logs will be saved to: {self.log_dir} ---")
        
        self.mse_history = []
        self.energy_history = []
        
    def update_stats(self, mse, energy):
        self.mse_history.append(mse)
        self.energy_history.append(energy)
        
    def plot_snapshot(self, epoch, model_embed, title_suffix=""):
        # 1. 数据准备
        weights = model_embed.weight.detach().cpu()
        weights_flat = torch.cat([weights.real, weights.imag], dim=1).numpy()
        
        # 2. PCA (线性投影)
        pca = PCA(n_components=2)
        coords_pca = pca.fit_transform(weights_flat)
        
        # 3. [New] t-SNE (非线性流形展开)
        # 仅在后期或 final 时运行 t-SNE (因为它很慢)
        use_tsne = (epoch > 1000) or ("FINAL" in title_suffix)
        coords_tsne = None
        if use_tsne:
            try:
                # perplexity 设为 30 或更小，适应小样本
                tsne = TSNE(n_components=2, perplexity=min(30, len(weights)-1), random_state=42, init='pca', learning_rate='auto')
                coords_tsne = tsne.fit_transform(weights_flat)
            except Exception as e:
                print(f"[Warn] t-SNE failed: {e}")

        # 4. 绘图 (调整为 2x2 布局 或 1x4 布局)
        plt.figure(figsize=(24, 6))
        
        # Subplot 1: Dynamics
        plt.subplot(1, 4, 1)
        plt.plot(self.mse_history, label='MSE', color='tab:blue', linewidth=1)
        plt.plot(self.energy_history, label='Energy', color='tab:orange', linestyle='--')
        plt.yscale('log')
        plt.title(f'Dynamics (Ep {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: PCA Topology
        plt.subplot(1, 4, 2)
        colors = np.arange(len(weights))
        plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=colors, cmap='turbo', s=20, alpha=0.8)
        plt.title('Linear Projection (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        # Subplot 3: [New] t-SNE Topology
        plt.subplot(1, 4, 3)
        if coords_tsne is not None:
            plt.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=colors, cmap='turbo', s=20, alpha=0.8)
            plt.title('Manifold Unfolding (t-SNE)')
        else:
            plt.text(0.5, 0.5, "t-SNE skipped (Early Epoch)", ha='center')
        
        # Subplot 4: Phase Hist
        plt.subplot(1, 4, 4)
        phases = weights.angle().numpy().flatten()
        plt.hist(phases, bins=50, color='purple', alpha=0.6, density=True)
        plt.title('Phase Dist')
        
        save_path = os.path.join(self.log_dir, f"epoch_{epoch:05d}{title_suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# ==========================================
#  Core Model (Manifold Learner)
# ==========================================
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

# ==========================================
#  Training Protocol
# ==========================================
def train_manifold(args):
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    
    # [Path Fix]: 传入当前脚本所在的目录 (Anla/)
    vis = ManifoldVisualizer(base_dir=_ANLA_ROOT)
    
    print(f"--- ANLA AGI: Manifold Evolution Protocol ---")
    print(f"--- Visualizer: ON | Target Epochs: {args.epochs} ---")
    
    model = AnlaManifoldModel(args.vocab_size, args.dim, args.layers, args.heads).to(device)
    
    # 1. 低熵初始化
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1: p.data.mul_(0.6)
            
    # 2. 构造拓扑学习任务 (N -> N+1)
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len)).to(device)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    
    print("Starting Evolution Loop...")
    start_time = time.time()
    converged = False 

    for epoch in range(args.epochs):
        # LR Schedule: 延长 Warmup 以适应更长的训练周期
        warmup_steps = 500
        if epoch < warmup_steps:
            current_lr = args.lr * (epoch + 1) / warmup_steps
        else:
            # 线性衰减
            progress = (epoch - warmup_steps) / (args.epochs - warmup_steps)
            current_lr = args.lr * (1.0 - progress * 0.9) # 最终保留 10% LR

        # === Forward ===
        z_pred = model.forward(input_ids)
        z_target = model.embed.forward(target_ids)
        
        # === Physics Engine ===
        force_vector = z_pred - z_target
        
        # 统计
        with torch.no_grad():
            mse_loss = 0.5 * (force_vector.abs().pow(2)).mean().item()
            avg_energy = model.embed.weight.abs().mean().item()
            vis.update_stats(mse_loss, avg_energy)

        # === Bidirectional Dynamics ===
        # Path A: Network Update
        model.manual_backward(force_vector, current_lr, args.weight_decay)
        
        # Path B: Target Evolution
        reaction_vector = -force_vector
        model.embed.manual_backward_explicit(
            grad=reaction_vector,
            indices=target_ids,
            lr=current_lr,
            weight_decay=args.weight_decay
        )
        
        # === Visualization & Logging ===
        plot_interval = 100 if epoch < 1000 else 500
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:05d} | MSE: {mse_loss:.6f} | Energy: {avg_energy:.4f} | LR: {current_lr:.5f}")
        
        if epoch % plot_interval == 0:
            vis.plot_snapshot(epoch, model.embed)
            
            # 收敛判定
            if mse_loss < 0.001 and epoch > 1000:
                print("\n[Converged] Manifold stabilized successfully.")
                # [Fix] 立即保存最终状态并退出
                vis.plot_snapshot(epoch, model.embed, title_suffix="_FINAL")
                converged = True
                break
    # [Fix] 循环结束后，强制保存最后一帧 (如果是因为跑满 epochs 而非 converge 退出)
    if not converged:
        print("\n[Completed] Reached max epochs.")
        # 这里传入 args.epochs 表示最终状态
        vis.plot_snapshot(args.epochs, model.embed, title_suffix="_FINAL")

    print(f"Total Time: {time.time() - start_time:.2f}s")
    
    # === Inference Demo ===
    print("\n--- Inference: Manifold Navigation (k-NN) ---")
    query_vec = z_pred[0, 0].detach() 
    target_id = target_ids[0, 0].item()
    
    all_vocab = model.embed.weight.detach()
    diff = all_vocab - query_vec
    dists = diff.abs().pow(2).sum(dim=1) # L2 Distance Squared
    
    nearest_id = dists.argmin().item()
    nearest_dist = dists[nearest_id].item()
    
    print(f"True Target ID : {target_id}")
    print(f"Predicted ID   : {nearest_id}")
    print(f"Dist          : {nearest_dist:.4f}")

    if target_id == nearest_id:
        print("[SUCCESS] Concept aligned in manifold space.")
    else:
        print("[MISS] Concept drift.")
    
    # === [New] Full Validation ===
    print("\n--- Full Manifold Validation ---")
    model.eval()
    with torch.no_grad():
        # 1. 构造全量测试数据 (0 -> 1, 1 -> 2, ..., 199 -> 0)
        all_ids = torch.arange(args.vocab_size, device=device).unsqueeze(0) # (1, Vocab)
        # 为了适配 seq_len，我们需要复制一下或者简单地逐个预测
        # 这里做一个简单的全量过网
        
        # 构造 Input: (Vocab, 1) -> (Vocab, Seq) 
        # 为了简单，我们只测长度为 1 的上下文，或者构造 batch
        # 实际上我们之前的训练 seq_len=10，这里简单起见，我们构造一个覆盖所有转移的 batch
        
        val_input = torch.arange(args.vocab_size, device=device).unsqueeze(1) # (Vocab, 1)
        # 这里需要 padding 或者 repeat 到 seq_len，但这取决于 position encoding
        # 简单起见，我们直接用全词表作为输入序列
        
        # 正确做法：构造一个长序列包含所有 ID
        full_seq = torch.arange(args.vocab_size, device=device).unsqueeze(0) # (1, Vocab)
        target_seq = torch.roll(full_seq, shifts=-1, dims=1)
        
        z_pred_full = model.forward(full_seq) # (1, Vocab, Dim)
        z_target_full = model.embed.forward(target_seq) # (1, Vocab, Dim)
        
        # 计算准确率
        correct_count = 0
        all_vocab_emb = model.embed.weight.detach() # (Vocab, Dim)
        
        for i in range(args.vocab_size):
            pred_vec = z_pred_full[0, i] # (Dim)
            true_id = target_seq[0, i].item()
            
            # k-NN 搜索
            dists = (all_vocab_emb - pred_vec).abs().pow(2).sum(dim=1)
            pred_id = dists.argmin().item()
            
            if pred_id == true_id:
                correct_count += 1
                
        acc = correct_count / args.vocab_size * 100
        print(f"Global Top-1 Accuracy: {acc:.2f}% ({correct_count}/{args.vocab_size})")
        
        if acc > 99.0:
            print("[PERFECT] The manifold is topologically isomorphic to the sequence.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [Mod] 默认 Epochs 提升至 5000 以观察完整的拓扑展开
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
