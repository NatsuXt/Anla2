import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from Anla.layers.linear import ComplexLinear
from Anla.layers.activation import PhaseTwist
from Anla.layers.normalization import ComplexRMSNorm

def run_xor_test():
    # --- 配置 ---
    # 这是一个 2D -> 1D 的分类问题（但在复数看来是 1D -> 1D）
    # Input: 1个复数 z
    # Output: 1个复数 (趋向 1 或 -1)
    
    HIDDEN_DIM = 64 # 给足宽度以形成复杂的决策面
    BATCH_SIZE = 128
    EPOCHS = 3000
    LEARNING_RATE = 0.005 # XOR 需要较强的扭曲，LR 稍大一点
    WEIGHT_DECAY = 1e-4
    
    device = torch.device('cpu')
    
    print("Running Complex XOR (Quadrant) Test...")
    
    # --- 模型 ---
    # 3层 MLP 结构: Linear -> Norm -> Act -> Linear -> Norm -> Act -> Linear
    class XORNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = ComplexLinear(1, HIDDEN_DIM)
            self.n1 = ComplexRMSNorm(HIDDEN_DIM)
            self.a1 = PhaseTwist(HIDDEN_DIM, init_gamma=0.05)
            
            self.l2 = ComplexLinear(HIDDEN_DIM, HIDDEN_DIM)
            self.n2 = ComplexRMSNorm(HIDDEN_DIM)
            self.a2 = PhaseTwist(HIDDEN_DIM, init_gamma=0.05)
            
            self.l3 = ComplexLinear(HIDDEN_DIM, 1)

        def forward(self, z):
            h1 = self.a1.forward(self.n1.forward(self.l1.forward(z)))
            h2 = self.a2.forward(self.n2.forward(self.l2.forward(h1)))
            out = self.l3.forward(h2)
            return out
            
        def manual_backward(self, delta, lr, wd):
            grad_h2 = self.l3.manual_backward(delta, lr, wd)
            
            grad_n2 = self.a2.manual_backward(grad_h2, lr)
            grad_l2 = self.n2.manual_backward(grad_n2, lr)
            grad_h1 = self.l2.manual_backward(grad_l2, lr, wd)
            
            grad_n1 = self.a1.manual_backward(grad_h1, lr)
            grad_l1 = self.n1.manual_backward(grad_n1, lr)
            self.l1.manual_backward(grad_l1, lr, wd)

    model = XORNet().to(device)
    
    loss_history = []
    acc_history = []
    
    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        # 1. 数据生成
        # 随机生成 z，模长 [0.5, 1.5]，相位 [-pi, pi]
        r = torch.rand(BATCH_SIZE, 1) + 0.5
        theta = torch.rand(BATCH_SIZE, 1) * 2 * np.pi - np.pi
        z_in = torch.polar(r, theta).to(device)
        
        # 2. 标签生成 (XOR 逻辑)
        # Q1 (0~pi/2) & Q3 (-pi~-pi/2) -> Class A (1)
        # Q2 (pi/2~pi) & Q4 (-pi/2~0) -> Class B (-1)
        # 用 sin(2*theta) 判断象限: 
        # sin(2theta) > 0 对应 Q1, Q3
        # sin(2theta) < 0 对应 Q2, Q4
        labels = torch.sign(torch.sin(2 * theta))
        # 转换为目标复数: 1+0i 或 -1+0i
        z_target = labels.type(torch.complex64)
        
        # 3. 前向与误差
        z_pred = model.forward(z_in)
        delta = z_target - z_pred
        
        loss = 0.5 * torch.mean(torch.abs(delta)**2).item()
        loss_history.append(loss)
        
        # 4. 反向传播
        model.manual_backward(delta, LEARNING_RATE, WEIGHT_DECAY)
        
        # 5. 计算准确率 (分类边界: 实部 > 0 为 A，< 0 为 B)
        pred_labels = torch.sign(z_pred.real)
        acc = (pred_labels == labels).float().mean().item()
        acc_history.append(acc)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    print(f"Final Acc: {acc:.2%}")

    # --- 可视化决策边界 ---
    if 'DISPLAY' in os.environ or os.name == 'nt':
        # 生成密集网格
        grid_x = torch.linspace(-1.5, 1.5, 100)
        grid_y = torch.linspace(-1.5, 1.5, 100)
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_z = torch.complex(xx, yy).flatten().unsqueeze(1).to(device)
        
        # 预测
        with torch.no_grad():
            grid_out = model.forward(grid_z)
            # 取实部作为决策依据
            decision = grid_out.real.reshape(100, 100).numpy()
            
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, label='Loss')
        plt.plot(acc_history, label='Accuracy')
        plt.title("XOR Training Process")
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.imshow(decision, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower', cmap='RdBu', alpha=0.8)
        plt.colorbar(label='Output Real Part')
        plt.title("Decision Boundary (Q1/Q3 vs Q2/Q4)")
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        # 画坐标轴
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_xor_test()
