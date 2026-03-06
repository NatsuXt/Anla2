import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 导入核心组件
from Anla.layers.linear import ComplexLinear
from Anla.layers.activation import PhaseTwist
from Anla.layers.normalization import ComplexRMSNorm

def run_z_square_test():
    # --- 1. 配置 ---
    # 这次我们不需要 Embedding，直接处理复数向量
    # 为了验证逼近能力，我们用一个宽一点的隐藏层
    INPUT_DIM = 1
    HIDDEN_DIM = 32
    OUTPUT_DIM = 1
    
    BATCH_SIZE = 128
    EPOCHS = 2000
    LEARNING_RATE = 0.002 # 使用我们验证过的“黄金学习率”
    WEIGHT_DECAY = 1e-4
    
    device = torch.device('cpu') # 继续用 CPU
    
    print(f"Running Z^2 Fitting Test on {device}...")
    
    # --- 2. 构建模型 ---
    # 结构: Input(1) -> Linear -> Norm -> Act -> Linear -> Norm -> Act -> Linear -> Output(1)
    # 我们用两层隐藏层给它足够的变换空间
    
    class ZSquareNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = ComplexLinear(INPUT_DIM, HIDDEN_DIM)
            self.n1 = ComplexRMSNorm(HIDDEN_DIM)
            self.a1 = PhaseTwist(HIDDEN_DIM, init_gamma=0.1) # 初始给大一点的 Gamma
            
            self.l2 = ComplexLinear(HIDDEN_DIM, HIDDEN_DIM)
            self.n2 = ComplexRMSNorm(HIDDEN_DIM)
            self.a2 = PhaseTwist(HIDDEN_DIM, init_gamma=0.1)
            
            self.l3 = ComplexLinear(HIDDEN_DIM, OUTPUT_DIM)
            # 输出层通常不需要 Norm/Act，直接线性投影回结果
            
        def forward(self, z):
            # 前向传播需要手动管理
            # Block 1
            h1 = self.l1.forward(z)
            h1 = self.n1.forward(h1)
            h1 = self.a1.forward(h1)
            
            # Block 2
            h2 = self.l2.forward(h1)
            h2 = self.n2.forward(h2)
            h2 = self.a2.forward(h2)
            
            # Output
            out = self.l3.forward(h2)
            return out
            
        def manual_backward(self, delta_out, lr, wd):
            # 反向传播链
            grad_h2 = self.l3.manual_backward(delta_out, lr, weight_decay=wd)
            
            grad_n2 = self.a2.manual_backward(grad_h2, lr)
            grad_l2 = self.n2.manual_backward(grad_n2, lr)
            grad_h1 = self.l2.manual_backward(grad_l2, lr, weight_decay=wd)
            
            grad_n1 = self.a1.manual_backward(grad_h1, lr)
            grad_l1 = self.n1.manual_backward(grad_n1, lr)
            grad_in = self.l1.manual_backward(grad_l1, lr, weight_decay=wd)
            
            return grad_in

    model = ZSquareNet().to(device)
    
    # --- 3. 训练循环 ---
    loss_history = []
    
    for epoch in range(EPOCHS):
        # A. 生成数据
        # 在单位圆内均匀采样 z = r * e^(i*theta)
        # 模长 r in [0, 1.5], 相位 theta in [-pi, pi]
        r = torch.rand(BATCH_SIZE, 1) * 1.5
        theta = torch.rand(BATCH_SIZE, 1) * 2 * np.pi - np.pi
        z_in = torch.polar(r, theta).to(device)
        
        # 目标: z^2
        z_target = z_in ** 2
        
        # B. Forward
        z_pred = model.forward(z_in)
        
        # C. Loss
        delta = z_target - z_pred
        loss = 0.5 * torch.mean(torch.abs(delta)**2).item()
        loss_history.append(loss)
        
        # D. Backward
        model.manual_backward(delta, LEARNING_RATE, WEIGHT_DECAY)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")

    # --- 4. 验证与可视化 ---
    print(f"Final Loss: {loss:.6f}")
    
    # 生成测试集 (规则网格) 用于可视化
    grid_r = torch.linspace(0, 1.5, 20)
    grid_theta = torch.linspace(-np.pi, np.pi, 40)
    mesh_r, mesh_theta = torch.meshgrid(grid_r, grid_theta, indexing='ij')
    test_z = torch.polar(mesh_r, mesh_theta).flatten().unsqueeze(1).to(device)
    
    pred_z = model.forward(test_z).detach()
    true_z = test_z ** 2
    
    # 计算相位误差
    phase_diff = torch.angle(pred_z) - torch.angle(true_z)
    # 归一化到 [-pi, pi]
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    avg_phase_error = torch.mean(torch.abs(phase_diff)).item()
    print(f"Avg Phase Error: {avg_phase_error:.4f} rad")
    
    # 绘图
    if 'DISPLAY' in os.environ or os.name == 'nt':
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_history)
        plt.title("Z^2 Fitting Loss")
        plt.yscale('log')
        plt.grid(True)
        
        # 可视化相位拟合效果
        # 我们取一个固定半径 r=1 的圆周，看相位是否倍增
        circle_z = torch.polar(torch.ones(100,1), torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)).to(device)
        circle_pred = model.forward(circle_z).detach()
        
        plt.subplot(1, 2, 2)
        plt.plot(torch.angle(circle_z).numpy(), torch.angle(circle_pred).numpy(), '.', label='Predicted')
        plt.plot(torch.angle(circle_z).numpy(), torch.angle(circle_z**2).numpy(), '-', alpha=0.5, label='True Z^2')
        plt.title("Phase Mapping (r=1)")
        plt.xlabel("Input Phase")
        plt.ylabel("Output Phase")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_z_square_test()
