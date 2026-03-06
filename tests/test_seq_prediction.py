import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.linear import ComplexLinear
from Anla.layers.positional import ComplexRotaryEmbedding

save_dir = os.path.join(project_root, 'Anla', 'Logs', 'photoshots')
os.makedirs(save_dir, exist_ok=True)

class AnlaTransformer(nn.Module):
    def __init__(self, vocab_size, dim, heads=4, num_layers=2):
        super().__init__()
        self.embed = ComplexEmbedding(vocab_size, dim)
        # [NEW] Positional Encoding
        self.pos_enc = ComplexRotaryEmbedding(dim, max_seq_len=512)
        self.layers = nn.ModuleList([
            ComplexTransformerBlock(dim, num_heads=heads) 
            for _ in range(num_layers)
        ])
        self.head = ComplexLinear(dim, vocab_size, mode='descent')

    def forward(self, x):
        h = self.embed.forward(x)
        h = self.pos_enc.forward(h) # Inject Position
        for layer in self.layers:
            h = layer.forward(h)
        out = self.head.forward(h)
        return out
    
    def manual_backward(self, grad_output, lr, weight_decay):
        grad_flow = self.head.manual_backward(grad_output, lr, weight_decay)
        for layer in reversed(self.layers):
            grad_flow = layer.manual_backward(grad_flow, lr, weight_decay)
        
        # [NEW] Backward through Position
        grad_flow = self.pos_enc.manual_backward(grad_flow)
        
        self.embed.manual_backward(grad_flow, lr, weight_decay)

def visualize_prediction_analysis(loss_hist, acc_hist, logits, target_ids, run_id="gpu_rotary"):
    print(f"\n[Visualizing] Generating report for: {run_id}...")
    plt.figure(figsize=(20, 10))
    
    # Dynamics
    plt.subplot(2, 2, 1)
    ax1 = plt.gca()
    ax1.plot(loss_hist, color='navy', label='Loss')
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.plot(acc_hist, color='orange', label='Acc')
    ax2.set_ylim(0, 105)
    plt.title(f'Dynamics (Final Acc: {acc_hist[-1]:.1f}%)')
    
    # Heatmap
    sample_idx = 0
    logits_mag = logits[sample_idx].abs().detach().cpu().numpy().T
    target_seq = target_ids[sample_idx].cpu().numpy()
    
    plt.subplot(2, 2, 2)
    plt.imshow(logits_mag, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar()
    for t, true_id in enumerate(target_seq):
        plt.scatter(t, true_id, color='red', marker='x', s=100)
    plt.title('Prediction Heatmap')
    
    # Errors
    pred_seq = logits[sample_idx].abs().argmax(dim=-1).cpu().numpy()
    errors = np.where(pred_seq != target_seq)[0]
    
    plt.subplot(2, 2, 3)
    plt.bar(range(len(pred_seq)), logits_mag.max(axis=0), color='skyblue')
    if len(errors) > 0:
        plt.bar(errors, logits_mag.max(axis=0)[errors], color='red')
    plt.title(f'Confidence & Errors (Count: {len(errors)})')
    
    # Phase
    logits_phase = logits[sample_idx].angle().detach().cpu().numpy().T
    plt.subplot(2, 2, 4)
    plt.imshow(logits_phase, aspect='auto', cmap='twilight', origin='lower')
    plt.title('Output Phase Distribution')
    
    save_path = os.path.join(save_dir, f'seq_pred_{run_id}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")

def test_sequence_prediction():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"=== Project Anla: GPU Rotary Test ({torch.cuda.get_device_name(0)}) ===")
    else:
        device = torch.device('cpu')
    
    # 配置
    vocab_size = 20
    dim = 128          
    num_layers = 2    
    seq_len = 12
    batch_size = 64
    
    target_lr = 0.001
    epochs = 5000 
    weight_decay = 1e-4 
    warmup_steps = 200
    
    print(f"Config: Dim={dim}, Layers={num_layers}, LR={target_lr}, With Rotary Positional Encoding")
    
    model = AnlaTransformer(vocab_size, dim, num_layers=num_layers).to(device)
    
    # Init Scaling
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1: p.data.mul_(0.6) 

    # Data
    raw_data = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    input_ids = raw_data
    target_ids = torch.roll(raw_data, shifts=-1, dims=1)
    
    loss_history = []
    acc_history = []
    
    print(f"Starting Training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_steps:
            lr = target_lr * (epoch + 1) / warmup_steps
        else:
            lr = target_lr
            
        # Forward
        logits = model.forward(input_ids)
        
        # Target
        with torch.no_grad():
            target_vectors = torch.zeros_like(logits)
            idx = target_ids.unsqueeze(-1)
            target_vectors.scatter_(2, idx, 1.0)
            
        # Loss
        diff = logits - target_vectors
        loss = 0.5 * (diff.abs().pow(2)).mean()
        loss_history.append(loss.item())
        
        # Acc
        with torch.no_grad():
            pred_ids = logits.abs().argmax(dim=-1)
            acc = (pred_ids == target_ids).float().mean().item() * 100
            acc_history.append(acc)
        
        if np.isnan(loss.item()):
            print(f"!!! NaN detected at epoch {epoch} !!!")
            break

        model.manual_backward(diff, lr=lr, weight_decay=weight_decay)
        
        if epoch % 500 == 0:
            elapsed = time.time() - start_time
            speed = (epoch + 1) / (elapsed + 1e-9)
            print(f"Epoch {epoch:05d} | Loss: {loss.item():.6f} | Acc: {acc:.1f}% | Speed: {speed:.1f} it/s")

    total_time = time.time() - start_time
    print(f"\nTotal Time: {total_time:.2f}s")
    
    visualize_prediction_analysis(loss_history, acc_history, logits, target_ids, run_id="gpu_rotary")

    print("\n=== Final Report ===")
    print(f"Final Acc: {acc_history[-1]:.2f}%")
    
    if acc_history[-1] > 99.0:
        print("\n[MISSION COMPLETE] 100% Convergence Achieved.")
        print("Project Anla now possesses both Physical Stability and Temporal Awareness.")
    else:
        print("\n[INFO] Improved, but check config.")

if __name__ == "__main__":
    test_sequence_prediction()
