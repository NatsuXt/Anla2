import torch
import time
import matplotlib.pyplot as plt

# 导入核心组件 (保持不变)
from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.linear import ComplexLinear
from Anla.layers.activation import PhaseTwist
from Anla.layers.normalization import ComplexRMSNorm

def run_pure_physics_test():
    # --- 0. 环境 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Anla Pure Test on: {device}")

    # --- 1. 物理参数修正 ---
    # [核心修正 A] 关闭耗散
    # 既然这个任务是强行记忆随机映射，初始信号很弱，
    # 我们不能让 Weight Decay 杀死微弱的萌芽。
    WEIGHT_DECAY = 0.0     
    
    VOCAB_SIZE = 100
    DIM = 64
    BATCH_SIZE = 32
    # 这是一个极其简单的任务，不需要序列，只需要点对点映射
    # 为了验证组件能力，我们模拟最纯粹的 N -> N+1
    
    LEARNING_RATE = 0.005 # 适中的学习率
    EPOCHS = 2000
    
    # --- 2. 架构：最原始的堆叠 (No Residuals) ---
    # 如果组件是好的，它们应该能直接传导梯度
    print("Architecture: Embed -> [Lin->Norm->Act] x2 -> Loss")
    
    embed = ComplexEmbedding(VOCAB_SIZE, DIM).to(device)
    
    linear1 = ComplexLinear(DIM, DIM, mode='descent').to(device)
    norm1 = ComplexRMSNorm(DIM).to(device)
    act1 = PhaseTwist(DIM).to(device)
    
    linear2 = ComplexLinear(DIM, DIM, mode='descent').to(device)
    norm2 = ComplexRMSNorm(DIM).to(device)
    act2 = PhaseTwist(DIM).to(device)

    # --- 3. 数据 ---
    # 构造固定的映射任务：Input ID -> Target ID
    # 这是一个确定性任务，模型理应能学会
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 5)).to(device) # SeqLen=5
    target_ids = (input_ids + 1) % VOCAB_SIZE
    
    loss_history = []
    mag_history = []
    
    print("\nStarting Dynamics Simulation...")
    
    for epoch in range(EPOCHS):
        # === Forward Path ===
        # 1. Embedding
        x0 = embed.forward(input_ids)
        # Flatten for MLP: (Batch * Seq, Dim)
        x_flat = x0.view(-1, DIM)
        
        # 2. Layer 1
        z1 = linear1.forward(x_flat)
        n1 = norm1.forward(z1)
        a1 = act1.forward(n1)
        
        # 3. Layer 2
        z2 = linear2.forward(a1)
        n2 = norm2.forward(z2)
        a2 = act2.forward(n2)
        
        # === Output & Loss ===
        logits = a2
        
        # Target: 也是从 Embedding 取出的单位向量
        # 注意：这里我们detach，因为target是作为"地标"存在的
        target_vecs = embed.forward(target_ids).detach().view(-1, DIM)
        
        # [核心修正 B] 几何正确的梯度
        # 我们要让 Prediction 靠近 Target。
        # MSE Loss 的梯度方向是 (Pred - Target)。
        # Linear(mode='descent') 执行 W -= lr * grad。
        # 组合：W -= lr * (Pred - Target)  ==> W += lr * (Target - Pred)
        # 这是物理正确的吸引力。
        grad_out = logits - target_vecs
        
        # 计算 Loss 用于观察 (MSE)
        # Loss = 0.5 * |Pred - Target|^2
        current_loss = 0.5 * torch.mean((logits - target_vecs).abs().pow(2)).item()
        loss_history.append(current_loss)
        
        # === Backward Path (Manual) ===
        # 没有任何残差连接，全靠组件传导
        
        # Layer 2 Backward
        grad_n2 = act2.manual_backward(grad_out, LEARNING_RATE)
        grad_z2 = norm2.manual_backward(grad_n2, LEARNING_RATE)
        grad_a1 = linear2.manual_backward(grad_z2, LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Layer 1 Backward
        grad_n1 = act1.manual_backward(grad_a1, LEARNING_RATE)
        grad_z1 = norm1.manual_backward(grad_n1, LEARNING_RATE)
        grad_x0 = linear1.manual_backward(grad_z1, LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Embedding Backward
        # [核心修正 C] 闭环：必须更新 Embedding
        # 否则模型只能调整中间层去匹配两个随机向量，难度极大
        embed.manual_backward(grad_x0, LEARNING_RATE)
        
        # === Monitoring ===
        if epoch % 100 == 0:
            # 监控输出模长。如果死掉，mag -> 0。
            # 正常的 ComplexRMSNorm 应该把模长维持在 1.0 * Scale (初始1.0)
            mag = logits.abs().mean().item()
            mag_history.append(mag)
            print(f"Epoch {epoch:04d} | Loss: {current_loss:.6f} | Mag: {mag:.4f}")

    print(f"Final Loss: {current_loss:.6f}")
    
    # 验证是否真的学会了
    if current_loss < 0.1:
        print("\n[SUCCESS] The pure stack learned the mapping!")
    else:
        print("\n[FAILURE] Dynamics failed to converge.")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Loss (No Weight Decay)')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(mag_history)
    plt.title('Output Magnitude')
    plt.show()

if __name__ == "__main__":
    run_pure_physics_test()
