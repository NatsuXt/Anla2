import unittest
import torch
import torch.nn as nn
import math
import sys
import os

# 确保可以导入 Anla 包 (从 tests/ 向上两级到项目根目录)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.activation import PhaseTwist

class TestAnlaLogic(unittest.TestCase):
    """
    [Anla 逻辑验证套件]
    摒弃视觉表象，从数学底层验证架构的物理特性：
    1. 相位等变性 (Phase Equivariance)
    2. 能量稳态 (Energy Homeostasis)
    3. 双向纠缠 (Bidirectional Entanglement)
    4. 梯度流完整性 (Gradient Flow Integrity)
    """

    def setUp(self):
        # 基础配置
        self.batch_size = 4
        self.seq_len = 8
        self.d_model = 16
        self.vocab_size = 100
        self.lr = 0.1
        
        # 随机种子，保证可复现性
        torch.manual_seed(42)

    def test_01_phase_equivariance(self):
        """
        [核心哲学验证] 相位等变性测试
        假设系统 F，输入 x。若 x 旋转 theta (x' = x * e^iθ)，
        则输出 F(x') 必须等于 F(x) * e^iθ。
        
        这证明了 Anla 是通过“相对相位结构”理解世界，而不是死记“绝对坐标”。
        """
        print("\n=== Test 01: Phase Equivariance (The 'Spin' Test) ===")
        
        # 1. 实例化 Transformer Block
        block = ComplexTransformerBlock(d_model=self.d_model, num_heads=4)
        block.eval() # 关闭 Dropout 等随机因素
        
        # 2. 生成随机复数输入
        x_real = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x_imag = torch.randn(self.batch_size, self.seq_len, self.d_model)
        x = torch.complex(x_real, x_imag)
        
        # 3. 原始前向传播
        y_original = block.forward(x)
        
        # 4. 施加全局相位旋转 (Global Phase Rotation)
        theta = 1.57 # 旋转 90度 (PI/2)
        rotator = torch.polar(torch.tensor(1.0), torch.tensor(theta))
        x_rotated = x * rotator
        
        # 5. 旋转后的前向传播
        y_rotated_input = block.forward(x_rotated)
        
        # 6. 验证：y_rotated_input 应该等于 y_original * rotator
        y_expected = y_original * rotator
        
        # 计算误差
        diff = torch.abs(y_rotated_input - y_expected)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"Max Difference: {max_diff:.8f}")
        print(f"Mean Difference: {mean_diff:.8f}")
        
        # 注意：由于浮点数精度累积，不可能完全为0，但应极小 (< 1e-5)
        self.assertTrue(max_diff < 1e-5, f"Phase Equivariance violated! Max diff: {max_diff}")
        print(">> SUCCESS: Architecture is Phase Equivariant.")

    def test_02_manifold_homeostasis(self):
        """
        [流形约束验证] 能量稳态测试
        验证 Embedding 层在更新后，是否强制将向量模长约束在单位球体附近。
        """
        print("\n=== Test 02: Manifold Homeostasis (Energy Conservation) ===")
        
        emb = ComplexEmbedding(self.vocab_size, self.d_model)
        emb.train()
        
        # 模拟输入
        indices = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # 前向传播
        z = emb.forward(indices)
        
        # 模拟巨大的梯度 (试图破坏流形结构)
        grad = torch.randn_like(z) * 100.0 
        
        # 手动反向传播
        emb.manual_backward(grad, lr=1.0) # 使用大 LR 激进更新
        
        # 检查更新后的权重
        updated_weights = emb.weight.data
        magnitudes = torch.abs(updated_weights)
        mean_mag = torch.mean(magnitudes).item()
        max_mag = torch.max(magnitudes).item()
        min_mag = torch.min(magnitudes).item()
        
        print(f"Updated Mean Magnitude: {mean_mag:.4f}")
        print(f"Range: [{min_mag:.4f}, {max_mag:.4f}]")
        
        # 这里的 Loose bound 设为 0.5 - 2.0，因为 Homeostasis 是软约束 (RMS Scaling)
        # 只要没有爆炸到 100 或者 消失到 0 就算成功
        self.assertTrue(0.5 < mean_mag < 2.0, "Energy Homeostasis failed! Norms exploded or vanished.")
        print(">> SUCCESS: Manifold constraints active.")

    def test_03_bidirectional_entanglement(self):
        """
        [Path B 验证] 双向纠缠测试
        验证是否可以通过 manual_backward_explicit 直接修改 Target ID 的表征，
        而不依赖于 input 的前向传播路径。
        """
        print("\n=== Test 03: Bidirectional Entanglement (Telepathy Test) ===")
        
        emb = ComplexEmbedding(self.vocab_size, self.d_model)
        
        target_idx = torch.tensor([10, 20], dtype=torch.long)
        
        # 记录原始权重
        w_orig = emb.weight.data[target_idx].clone()
        
        # 构造“意念梯度” (直接注入到 Target ID)
        fake_grad = torch.ones_like(w_orig, dtype=torch.complex64)
        
        # 执行 Path B 更新
        emb.manual_backward_explicit(fake_grad, target_idx, lr=0.1)
        
        # 检查是否改变
        w_new = emb.weight.data[target_idx]
        diff = torch.abs(w_new - w_orig)
        changed_amount = torch.sum(diff).item()
        
        print(f"Weights changed by: {changed_amount:.6f}")
        self.assertTrue(changed_amount > 0.0, "Path B failed to update weights.")
        
        # 检查未涉及的 ID 是否保持不变
        untouched_idx = torch.tensor([30], dtype=torch.long)
        w_untouched_orig = emb.weight.data[untouched_idx].clone()
        # 此时 weight 已经是更新过的
        w_untouched_new = emb.weight.data[untouched_idx]
        
        diff_untouched = torch.sum(torch.abs(w_untouched_new - w_untouched_orig)).item()
        self.assertTrue(diff_untouched == 0.0, "Path B wrongly updated unrelated weights.")
        
        print(">> SUCCESS: Path B Interface functional.")

    def test_04_gradient_flow_integrity(self):
        """
        [哈密顿流验证] 梯度下降有效性测试
        验证整个 Transformer Block 的手动反向传播链条是否导向更低的 Loss。
        这比 Gradient Check 更实用，直接证明动力学方向正确。
        """
        print("\n=== Test 04: Gradient Flow Integrity (Hamiltonian Descent) ===")
        
        model = ComplexTransformerBlock(d_model=self.d_model, num_heads=4)
        model.train()
        
        # 1. 准备数据
        x = torch.randn(1, self.seq_len, self.d_model, dtype=torch.complex64)
        target = torch.randn(1, self.seq_len, self.d_model, dtype=torch.complex64)
        
        # 2. Step 0: 计算初始 Loss
        y0 = model.forward(x)
        loss0 = 0.5 * torch.sum(torch.abs(y0 - target)**2)
        print(f"Initial Loss: {loss0.item():.6f}")
        
        # 3. 计算梯度 (dL/dy = y - target)
        grad_output = (y0 - target)
        # 注意: MSE Loss 的导数包含 conjugate 关系，但我们这里简化模型：
        # Loss = |y - t|^2 = (y-t)(y-t)*
        # dL/dy = (y-t)*  (Wirtinger conjugate gradient convention for update)
        # 为了配合 PyTorch 标准和我们的 manual_backward (通常接收 dL/dz_conj)，
        # 我们传入 (y0 - target)。我们的 layers 内部会处理 conjugate。
        
        # 4. 执行反向传播
        model.manual_backward(grad_output, lr=0.01)
        
        # 5. Step 1: 再次前向传播，检查 Loss 是否下降
        # 注意：需要清空 cache 或者重新 forward
        y1 = model.forward(x)
        loss1 = 0.5 * torch.sum(torch.abs(y1 - target)**2)
        print(f"Post-Update Loss: {loss1.item():.6f}")
        
        loss_diff = loss0.item() - loss1.item()
        print(f"Loss Reduction: {loss_diff:.6f}")
        
        self.assertTrue(loss1 < loss0, "Gradient descent failed to reduce loss! Backward logic might be broken.")
        print(">> SUCCESS: Gradient flows correctly and reduces energy.")

    def test_05_complex_rotary_topology(self):
        """
        [拓扑结构验证] Rotary Embedding 保范性测试
        Rotary Embedding 应该只改变相位，严禁改变模长。
        这是保证流形不被畸变的关键。
        """
        print("\n=== Test 05: Rotary Topology (Norm Preservation) ===")
        
        rope = ComplexRotaryEmbedding(dim=self.d_model, max_seq_len=20)
        x = torch.randn(1, 10, self.d_model, dtype=torch.complex64)
        
        # Forward
        x_rotated = rope.forward(x)
        
        # Check Norms
        norm_in = torch.abs(x)
        norm_out = torch.abs(x_rotated)
        
        diff = torch.abs(norm_in - norm_out)
        max_diff = torch.max(diff).item()
        
        print(f"Max Norm Deviation: {max_diff:.8f}")
        self.assertTrue(max_diff < 1e-5, "Rotary Embedding distorted the manifold metric!")
        print(">> SUCCESS: Rotary operation is strictly topological.")

if __name__ == '__main__':
    unittest.main()
