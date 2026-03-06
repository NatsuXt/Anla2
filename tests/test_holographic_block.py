import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 路径与环境
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# 图片保存路径
save_dir = os.path.join(project_root, 'Anla', 'Logs', 'photoshots')
os.makedirs(save_dir, exist_ok=True)

try:
    from Anla.layers.transformer_block import ComplexTransformerBlock
    from Anla.utils.complex_ops import create_complex_tensor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ==========================================
# 2. 可视化函数 (保持不变)
# ==========================================
def visualize_analysis(loss_history, model, run_id="latest"):
    print(f"\n[Visualizing] Generating report for run: {run_id}...")
    
    plt.figure(figsize=(18, 6))
    
    # --- 图 1: Loss ---
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label='MSE Loss', color='navy')
    plt.yscale('log')
    plt.title(f'Loss Convergence (Final: {loss_history[-1]:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    # --- 检查缓存 ---
    if model.attn.attn_cache is not None:
        attn_batch = model.attn.attn_cache[0].detach().cpu()
        
        # --- 图 2: Attention Magnitude ---
        plt.subplot(1, 3, 2)
        # 取平均幅值
        avg_attn_mag = attn_batch.abs().mean(dim=0)
        
        im = plt.imshow(avg_attn_mag, cmap='inferno', interpolation='nearest', vmin=0, vmax=1.0)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f'Avg Attention Magnitude\n(Target: Anti-Diagonal /)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # --- 图 3: Phase Shift ---
        plt.subplot(1, 3, 3)
        head0_phase = attn_batch[0].angle()
        im_phase = plt.imshow(head0_phase, cmap='twilight', interpolation='nearest')
        plt.colorbar(im_phase, fraction=0.046, pad=0.04)
        plt.title('Head 0 Phase Shift\n(-pi to pi)')
        plt.xlabel('Key')
        plt.ylabel('Query')
    else:
        print("Warning: attn_cache is None! Skipping Attention plots.")
        plt.subplot(1, 3, 2)
        plt.text(0.5, 0.5, 'Cache Empty', ha='center')
    
    save_path = os.path.join(save_dir, f'holographic_test_{run_id}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Saved] Visualization saved to: {save_path}")
    plt.close()

# ==========================================
# 3. 训练主逻辑
# ==========================================
def test_holographic_learning():
    print("=== Testing Holographic Transformer Block (Visualized) ===")
    
    # 配置
    d_model = 16
    seq_len = 8
    batch_size = 16
    heads = 4
    
    lr = 0.02
    epochs = 2000 # 您可以改回 10000，但演示用 2000 足够
    device = 'cpu'
    
    print(f"Initializing Model...")
    model = ComplexTransformerBlock(d_model, num_heads=heads).to(device)
    
    input_data = create_complex_tensor((batch_size, seq_len, d_model), device=device)
    target_data = torch.flip(input_data, dims=[1])
    
    print(f"Starting Training ({epochs} epochs)...")
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward
        pred = model.forward(input_data)
        
        # Loss
        diff = pred - target_data
        loss = 0.5 * (diff.abs().pow(2)).mean()
        loss_history.append(loss.item())
        
        # Backward (这步会清空缓存!)
        grad_out = diff
        
        grad_norm = grad_out.abs().max()
        if grad_norm > 1.0:
            grad_out = grad_out / grad_norm
            
        model.manual_backward(grad_out, lr=lr)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")

    # ==========================================
    # [关键修复] 最后再跑一次 Forward 以填充缓存
    # ==========================================
    print("\n[Finalizing] Re-running forward pass to capture attention map...")
    with torch.no_grad():
        # 必须确保 training=True，因为 Anla 仅在训练模式下缓存 attention
        model.train() 
        _ = model.forward(input_data)
        # 注意：这里不再调用 backward，所以缓存会被保留下来供绘图使用

    # 可视化
    visualize_analysis(loss_history, model, run_id="reverse_task_fixed")

    print("\n=== Summary ===")
    print(f"Final Loss: {loss_history[-1]:.6f}")

if __name__ == "__main__":
    test_holographic_learning()
