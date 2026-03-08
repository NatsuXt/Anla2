"""
保存位置: Anla/experiments/nlp/byte_mlm_v7.py

Byte-Level Masked Language Model — 复数 vs 实数 Transformer 对照实验
=========================================================================

目的:
    在真实 NLP 数据上检验 Anla 复数 Transformer 的语义学习能力,
    并与等架构的实数 Transformer 进行公平对照。

    Ring Inpainting 验证了复数计算链路的正确性和可训练性,
    但无法回答: "复数架构能否学到有意义的语义表示?"
    本实验直接回答这个问题。

任务:
    Byte-level MLM — 从真实文本中采样 byte 序列 (长度 S),
    随机 mask 15% 的位置, 预测被 mask 的 byte。
    V=256 (byte 词表) 恰好与 Config B 一致, 模型架构零修改。

模型:
    [1] complex — Anla v7 复数 Transformer (直接输出, 无 Encoder/Decoder)
        · 与 capacity_pressure_test_v7.py 中的 AnlaManifoldInpainter_v7 完全一致
        · Boltzmann-Elegant 损失 + 手动反向传播 + Polar Adam
        · D=64 复数维度 (= 128 实数自由度/token)

    [2] real — 标准实数 Transformer (PyTorch 内置组件)
        · Pre-Norm + RoPE + Weight Tying + Cross-Entropy
        · d_model=128 (匹配复数模型的信息维度: 64 复数 = 128 实数)
        · 标准 AdamW 优化器
        · 参数量会略多于复数模型 (因为实数权重矩阵是 128×128
          而非 64×64 复数), 但信息容量匹配

数据:
    默认自动下载 tiny_shakespeare (~1.1MB)。
    也可通过 --data-path 指定任意 UTF-8 文本文件。

用法:
    # 复数模型 (默认)
    python -m Anla.experiments.nlp.byte_mlm_v7 --model complex

    # 实数基线
    python -m Anla.experiments.nlp.byte_mlm_v7 --model real

    # 自定义数据
    python -m Anla.experiments.nlp.byte_mlm_v7 --data-path /path/to/text.txt

    # 对照运行 (两个模型都跑)
    python -m Anla.experiments.nlp.byte_mlm_v7 --model both
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# [Path Fix]
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

# 复数模型组件 (仅在 --model complex 时使用)
from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.losses.boltzmann_elegant import compute_boltzmann_elegant_loss_and_force


# =====================================================================
#  数据: Byte-Level Text Generator
# =====================================================================

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(cache_dir: str) -> str:
    """下载 tiny_shakespeare 数据集 (~1.1MB)。"""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tiny_shakespeare.txt")
    if os.path.exists(path):
        print(f"  [数据] 使用缓存: {path}")
        return path
    print(f"  [数据] 下载 tiny_shakespeare...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    size = os.path.getsize(path)
    print(f"  [数据] 完成: {size:,} bytes → {path}")
    return path


class TextByteGenerator:
    """
    Byte-level MLM 数据生成器。

    从文本文件中读取原始 bytes (0–255),
    按位置切分为 train/test,
    采样固定长度的 byte 序列并应用 BERT-style masking。

    与 RingDataGenerator 的接口完全兼容:
        generate_train_batch(batch_size, mask_mode, mask_prob, max_span)
        generate_test_batch(batch_size, mask_mode, mask_prob, max_span)
    """

    def __init__(self, data_path: str, seq_len: int, mask_id: int,
                 test_frac: float = 0.1, seed: int = 42):
        """
        Args:
            data_path:  UTF-8 文本文件路径
            seq_len:    序列长度 (bytes)
            mask_id:    MASK token ID (= 256, 超出 byte 范围)
            test_frac:  测试集比例 (按字节位置划分)
            seed:       随机种子
        """
        self.seq_len = seq_len
        self.mask_id = mask_id

        # 读取原始 bytes
        with open(data_path, 'rb') as f:
            raw = f.read()
        self.data = np.frombuffer(raw, dtype=np.uint8)
        total_len = len(self.data)

        # 按位置划分 train/test (避免数据泄漏)
        split_point = int(total_len * (1.0 - test_frac))
        self.train_data = self.data[:split_point]
        self.test_data = self.data[split_point:]

        print(f"  [数据] 总 bytes: {total_len:,}")
        print(f"  [数据] 训练: {len(self.train_data):,}, "
              f"测试: {len(self.test_data):,}")
        print(f"  [数据] 唯一 byte 值: "
              f"{len(np.unique(self.data))}/256")

    def _generate_batch(self, data: np.ndarray, batch_size: int,
                        mask_mode: str = 'bert',
                        mask_prob: float = 0.15,
                        max_span: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """从数据中采样 batch 并应用 masking。"""
        input_ids_list = []
        target_ids_list = []
        max_start = len(data) - self.seq_len

        for _ in range(batch_size):
            # 随机采样起始位置
            start = random.randint(0, max_start)
            seq = data[start: start + self.seq_len].copy()
            seq_tensor = torch.tensor(seq, dtype=torch.long)

            inp = seq_tensor.clone()
            tgt = torch.full_like(seq_tensor, -100)

            if mask_mode == 'bert':
                # BERT 风格: 每个位置独立 mask
                mask_flags = torch.rand(self.seq_len) < mask_prob
                if not mask_flags.any():
                    mask_flags[random.randint(0, self.seq_len - 1)] = True
                inp[mask_flags] = self.mask_id
                tgt[mask_flags] = seq_tensor[mask_flags]

            elif mask_mode == 'span':
                # Span 风格: 连续 mask
                mask_len = random.randint(1, max_span)
                mask_start = random.randint(0, self.seq_len - mask_len)
                inp[mask_start: mask_start + mask_len] = self.mask_id
                tgt[mask_start: mask_start + mask_len] = \
                    seq_tensor[mask_start: mask_start + mask_len]

            input_ids_list.append(inp)
            target_ids_list.append(tgt)

        return torch.stack(input_ids_list), torch.stack(target_ids_list)

    def generate_train_batch(self, batch_size: int,
                             mask_mode: str = 'bert',
                             mask_prob: float = 0.15,
                             max_span: int = 5):
        return self._generate_batch(self.train_data, batch_size,
                                    mask_mode, mask_prob, max_span)

    def generate_test_batch(self, batch_size: int,
                            mask_mode: str = 'bert',
                            mask_prob: float = 0.15,
                            max_span: int = 5):
        return self._generate_batch(self.test_data, batch_size,
                                    mask_mode, mask_prob, max_span)


# =====================================================================
#  复数模型: AnlaManifoldInpainter_v7 (与 capacity test 完全一致)
# =====================================================================

class ComplexByteMLM(nn.Module):
    """
    Anla v7 复数 Transformer — Byte-level MLM。
    与 capacity_pressure_test_v7 中的 AnlaManifoldInpainter_v7 完全一致。
    重新定义以避免跨实验脚本的脆弱导入。

    信号路径:
        input_ids → ComplexEmbedding(257, D) → ComplexRotary
        → TransformerBlock(×3) → z_pred (直接输出)
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 64,
                 num_heads: int = 4, num_blocks: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = ComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=512)
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads=num_heads, ff_mult=4)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)
        for block in self.blocks:
            z = block.forward(z, mask=None)
        return z

    def manual_backward(self, force: torch.Tensor, lr: float, wd: float):
        grad = force
        for block in reversed(self.blocks):
            grad = block.manual_backward(grad, lr, wd)
        grad = self.rotary.manual_backward(grad)
        self.embedding.manual_backward(grad, lr, weight_decay=0.0)


# =====================================================================
#  实数基线: 标准 PyTorch Transformer + Weight Tying
# =====================================================================

class RealRotaryEmbedding(nn.Module):
    """实数 RoPE 位置编码 (与 LLaMA 一致)。"""
    def __init__(self, d_model: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, D)"""
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (S, D/2)
        cos_f = freqs.cos().unsqueeze(0)  # (1, S, D/2)
        sin_f = freqs.sin().unsqueeze(0)  # (1, S, D/2)

        x1 = x[..., ::2]   # 偶数维
        x2 = x[..., 1::2]  # 奇数维
        out = torch.zeros_like(x)
        out[..., ::2] = x1 * cos_f - x2 * sin_f
        out[..., 1::2] = x1 * sin_f + x2 * cos_f
        return out


class RealByteMLM(nn.Module):
    """
    标准实数 Transformer — Byte-level MLM。

    使用 PyTorch 原生组件, Pre-Norm 架构, Weight Tying。
    d_model=128 匹配复数模型的 64 复数维度 (= 128 实数维度)。

    信号路径:
        input_ids → Embedding(257, 128) → RoPE
        → [Pre-Norm → MultiHeadAttn → Residual
           → Pre-Norm → FFN → Residual] × 3
        → LayerNorm → logits (= h @ W_embed^T, weight tying)
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 128,
                 num_heads: int = 4, num_blocks: int = 3, ff_mult: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding (输入和输出共享, weight tying)
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # RoPE
        self.rotary = RealRotaryEmbedding(d_model, max_seq_len=512)

        # Transformer Blocks (Pre-Norm)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(
                    d_model, num_heads, batch_first=True, dropout=0.0),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * ff_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ff_mult, d_model),
                ),
            }))

        # 输出 LayerNorm (LLaMA 风格)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S), long — input_ids
        Returns:
            logits: (B, S, V) — 分类 logits
        """
        # Embedding + RoPE
        h = self.embedding(x)
        h = self.rotary(h)

        # Transformer Blocks (Pre-Norm)
        for block in self.blocks:
            # Attention sub-layer
            residual = h
            h_norm = block['norm1'](h)
            attn_out, _ = block['attn'](h_norm, h_norm, h_norm,
                                        need_weights=False)
            h = residual + attn_out

            # FFN sub-layer
            residual = h
            h_norm = block['norm2'](h)
            ff_out = block['ff'](h_norm)
            h = residual + ff_out

        # Output: LayerNorm → Weight Tying
        h = self.output_norm(h)
        # logits = h @ embedding.weight[:vocab_size].T
        logits = F.linear(h, self.embedding.weight[:self.vocab_size])

        return logits


# =====================================================================
#  评估函数
# =====================================================================

@torch.no_grad()
def evaluate_complex_mlm(model, gen, batch_size, device,
                         num_batches=10, mask_mode='bert',
                         mask_prob=0.15):
    """评估复数模型: 最近邻解码准确率。"""
    model.eval()
    total_correct = 0
    total_count = 0

    all_embeds = model.embedding.weight.data[:model.vocab_size]

    for _ in range(num_batches):
        inp, tgt = gen.generate_test_batch(
            batch_size, mask_mode=mask_mode, mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)

        z_pred = model.forward(inp)
        valid = (tgt != -100)
        if not valid.any():
            continue

        z_masked = z_pred[valid]
        true_ids = tgt[valid]

        # 最近邻解码
        dists = (z_masked.unsqueeze(1) - all_embeds.unsqueeze(0)) \
            .abs().pow(2).sum(dim=-1)
        pred_ids = dists.argmin(dim=-1)

        total_correct += (pred_ids == true_ids).sum().item()
        total_count += true_ids.shape[0]

    acc = total_correct / max(total_count, 1)
    return acc


@torch.no_grad()
def evaluate_real_mlm(model, gen, batch_size, device,
                      num_batches=10, mask_mode='bert',
                      mask_prob=0.15):
    """评估实数模型: Cross-Entropy 准确率和 Perplexity。"""
    model.eval()
    total_correct = 0
    total_count = 0
    total_loss = 0.0

    for _ in range(num_batches):
        inp, tgt = gen.generate_test_batch(
            batch_size, mask_mode=mask_mode, mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)

        logits = model.forward(inp)
        valid = (tgt != -100)
        if not valid.any():
            continue

        logits_flat = logits[valid]
        tgt_flat = tgt[valid]

        loss = F.cross_entropy(logits_flat, tgt_flat, reduction='sum')
        pred_ids = logits_flat.argmax(dim=-1)

        total_loss += loss.item()
        total_correct += (pred_ids == tgt_flat).sum().item()
        total_count += tgt_flat.shape[0]

    acc = total_correct / max(total_count, 1)
    avg_loss = total_loss / max(total_count, 1)
    ppl = math.exp(min(avg_loss, 20.0))  # 截断防溢出
    return acc, ppl


# =====================================================================
#  训练: 复数模型 (手动反向传播, Boltzmann-Elegant)
# =====================================================================

def train_complex(model, gen, cfg, device, output_dir):
    """训练复数模型并返回结果。"""
    os.makedirs(output_dir, exist_ok=True)

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr_base = cfg['lr']
    warmup = cfg['warmup_epochs']
    wd = cfg['weight_decay']
    mask_prob = cfg['mask_prob']
    log_interval = cfg['log_interval']
    reaction_scale = cfg['reaction_scale']
    vocab_size = 256

    history = {
        'epochs': [], 'loss': [], 'train_acc': [], 'test_acc': [],
        'tau': [], 'p_target': [], 'neg_pct': [],
        'emb_rms': [], 'loss_elegant': [], 'lr': [],
    }

    best_train_acc = 0.0
    rho = 0.0
    rho_init = None
    rho_smooth = None
    rho_beta = 0.95

    t_start = time.time()

    for epoch in range(epochs):
        model.train()

        # Warmup LR
        lr = lr_base * min(1.0, max(1, epoch) / warmup)

        inp, tgt = gen.generate_train_batch(
            batch_size, mask_mode='bert', mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)
        valid = (tgt != -100)

        z_pred = model.forward(inp)

        all_embeds = model.embedding.weight.data[:vocab_size].detach()
        loss_val, force_a, force_b, info = \
            compute_boltzmann_elegant_loss_and_force(
                z_pred=z_pred, all_embeds=all_embeds,
                target_ids=tgt, valid_mask=valid)

        # Path A: 模型权重
        model.manual_backward(force_a, lr, wd)

        # Path B: Embedding
        reaction_lr = lr * reaction_scale * rho
        if reaction_lr > 1e-12 and valid.any():
            valid_ids = tgt[valid]
            valid_react = -force_b[valid]
            model.embedding.manual_backward_explicit(
                grad=valid_react, indices=valid_ids,
                lr=reaction_lr, weight_decay=0.0)

        # rho 更新
        rl = info.get('loss_elegant', loss_val)
        if rho_init is None:
            rho_init = rl
            rho_smooth = rl
        else:
            rho_smooth = rho_beta * rho_smooth + (1 - rho_beta) * rl
        rho = max(0, min(1, (rho_init - rho_smooth) / rho_init)) \
            if rho_init > 1e-12 else 0.0

        # 日志
        if epoch % log_interval == 0:
            # 训练准确率 (当前 batch)
            with torch.no_grad():
                z_m = z_pred[valid]
                t_m = tgt[valid]
                if t_m.numel() > 0:
                    d2 = (z_m.unsqueeze(1) - all_embeds.unsqueeze(0)) \
                        .abs().pow(2).sum(-1)
                    tr_acc = (d2.argmin(-1) == t_m).float().mean().item()
                else:
                    tr_acc = 0.0

            emb_rms = model.embedding.weight.data[:vocab_size] \
                .abs().pow(2).mean().sqrt().item()

            # 测试
            test_acc = -1.0
            if epoch % (log_interval * 5) == 0 or epoch == epochs - 1:
                test_acc = evaluate_complex_mlm(
                    model, gen, batch_size, device, num_batches=5)

            history['epochs'].append(epoch)
            history['loss'].append(loss_val)
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(test_acc)
            history['tau'].append(info.get('tau', 0))
            history['p_target'].append(info.get('p_target_mean', 0))
            history['neg_pct'].append(info.get('negative_margin_ratio', 0))
            history['emb_rms'].append(emb_rms)
            history['loss_elegant'].append(info.get('loss_elegant', 0))
            history['lr'].append(lr)

            if tr_acc > best_train_acc:
                best_train_acc = tr_acc
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'best.pth'))

            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""
            print(f"  [complex] Ep {epoch:05d} | "
                  f"L: {loss_val:.4f} | Tr: {tr_acc:.2%}{test_str}"
                  f" | τ: {info.get('tau',0):.3f}"
                  f" | p: {info.get('p_target_mean',0):.3f}"
                  f" | ρ: {rho:.3f}"
                  f" | RMS: {emb_rms:.3f}")

    elapsed = time.time() - t_start

    # 最终评估
    final_acc = evaluate_complex_mlm(
        model, gen, batch_size, device, num_batches=20)

    result = {
        'model': 'complex',
        'best_train_acc': best_train_acc,
        'final_test_acc': final_acc,
        'total_params': sum(p.numel() for p in model.parameters()),
        'training_time_sec': elapsed,
        'history': history,
        'config': cfg,
    }

    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  [complex] 完成: Train {best_train_acc:.2%}, "
          f"Test {final_acc:.2%}, {elapsed:.0f}s")
    return result


# =====================================================================
#  训练: 实数模型 (标准 autograd, Cross-Entropy)
# =====================================================================

def train_real(model, gen, cfg, device, output_dir):
    """训练实数模型并返回结果。"""
    os.makedirs(output_dir, exist_ok=True)

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr_base = cfg['lr']
    warmup = cfg['warmup_epochs']
    wd = cfg['weight_decay']
    mask_prob = cfg['mask_prob']
    log_interval = cfg['log_interval']
    vocab_size = 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base,
                                  weight_decay=wd, betas=(0.9, 0.99))

    # Warmup + Cosine Decay scheduler
    def lr_lambda(step):
        if step < warmup:
            return max(1, step) / warmup
        return 1.0  # 固定 lr (与复数模型一致)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        'epochs': [], 'loss': [], 'train_acc': [], 'test_acc': [],
        'test_ppl': [], 'lr': [],
    }

    best_train_acc = 0.0
    t_start = time.time()

    for epoch in range(epochs):
        model.train()

        inp, tgt = gen.generate_train_batch(
            batch_size, mask_mode='bert', mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)
        valid = (tgt != -100)

        logits = model.forward(inp)  # (B, S, V)

        # Cross-Entropy (仅在 mask 位置计算)
        if valid.any():
            loss = F.cross_entropy(logits[valid], tgt[valid])
        else:
            loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 日志
        if epoch % log_interval == 0:
            with torch.no_grad():
                if valid.any():
                    tr_acc = (logits[valid].argmax(-1) == tgt[valid]) \
                        .float().mean().item()
                else:
                    tr_acc = 0.0

            test_acc, test_ppl = -1.0, -1.0
            if epoch % (log_interval * 5) == 0 or epoch == epochs - 1:
                test_acc, test_ppl = evaluate_real_mlm(
                    model, gen, batch_size, device, num_batches=5)

            history['epochs'].append(epoch)
            history['loss'].append(loss.item())
            history['train_acc'].append(tr_acc)
            history['test_acc'].append(test_acc)
            history['test_ppl'].append(test_ppl)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            if tr_acc > best_train_acc:
                best_train_acc = tr_acc
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'best.pth'))

            test_str = f" | Test: {test_acc:.2%}" if test_acc >= 0 else ""
            ppl_str = f" | PPL: {test_ppl:.1f}" if test_ppl >= 0 else ""
            print(f"  [real] Ep {epoch:05d} | "
                  f"L: {loss.item():.4f} | Tr: {tr_acc:.2%}{test_str}"
                  f"{ppl_str}")

    elapsed = time.time() - t_start

    final_acc, final_ppl = evaluate_real_mlm(
        model, gen, batch_size, device, num_batches=20)

    result = {
        'model': 'real',
        'best_train_acc': best_train_acc,
        'final_test_acc': final_acc,
        'final_test_ppl': final_ppl,
        'total_params': sum(p.numel() for p in model.parameters()),
        'training_time_sec': elapsed,
        'history': history,
        'config': cfg,
    }

    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  [real] 完成: Train {best_train_acc:.2%}, "
          f"Test {final_acc:.2%}, PPL {final_ppl:.1f}, {elapsed:.0f}s")
    return result


# =====================================================================
#  主入口
# =====================================================================

# 默认配置 (与 Config B 对齐, 适配 NLP)
DEFAULT_CFG = {
    'd_model_complex': 64,       # 复数模型维度
    'd_model_real': 128,         # 实数模型维度 (64 复数 = 128 实数)
    'num_heads': 4,
    'num_blocks': 3,
    'seq_len': 64,               # 比 Ring 的 32 长一倍, NLP 需要更多上下文
    'batch_size': 64,            # NLP 数据量大, 可用更大 batch
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 15000,
    'warmup_epochs': 500,
    'mask_prob': 0.15,
    'reaction_scale': 0.1,
    'log_interval': 200,
}


def main():
    parser = argparse.ArgumentParser(
        description="Byte-level MLM: 复数 vs 实数 Transformer 对照实验"
    )
    parser.add_argument(
        '--model', type=str, default='complex',
        choices=['complex', 'real', 'both'],
        help='模型类型: complex (Anla v7), real (标准 Transformer), both (对照)'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='文本文件路径 (默认: 自动下载 tiny_shakespeare)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='覆盖默认 epoch 数'
    )
    parser.add_argument(
        '--seq-len', type=int, default=None,
        help='覆盖默认序列长度'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='覆盖默认 batch 大小'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_ANLA_ROOT, 'Logs', 'nlp_byte_mlm_v7'),
        help='输出目录'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='计算设备: auto/cpu/cuda'
    )
    args = parser.parse_args()

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 配置
    cfg = DEFAULT_CFG.copy()
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.seq_len is not None:
        cfg['seq_len'] = args.seq_len
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size

    # 数据
    if args.data_path is None:
        cache_dir = os.path.join(_ANLA_ROOT, 'data')
        data_path = download_tiny_shakespeare(cache_dir)
    else:
        data_path = args.data_path

    mask_id = 256  # 超出 byte 范围
    gen = TextByteGenerator(
        data_path=data_path,
        seq_len=cfg['seq_len'],
        mask_id=mask_id,
        test_frac=0.1,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # ---- 复数模型 ----
    if args.model in ('complex', 'both'):
        print("\n" + "=" * 72)
        print("  复数模型 (Anla v7 — Boltzmann-Elegant)")
        print(f"  d_model={cfg['d_model_complex']} (complex), "
              f"blocks={cfg['num_blocks']}, heads={cfg['num_heads']}")
        print("=" * 72)

        model_c = ComplexByteMLM(
            vocab_size=256,
            d_model=cfg['d_model_complex'],
            num_heads=cfg['num_heads'],
            num_blocks=cfg['num_blocks'],
        ).to(device)

        n_params = sum(p.numel() for p in model_c.parameters())
        print(f"  参数量: {n_params:,} (复数元素)")
        # 复数参数的实数等效自由度
        n_real_equiv = sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in model_c.parameters()
        )
        print(f"  实数等效自由度: {n_real_equiv:,}")

        out_c = os.path.join(args.output_dir, 'complex')
        results['complex'] = train_complex(
            model_c, gen, cfg, device, out_c)

    # ---- 实数模型 ----
    if args.model in ('real', 'both'):
        print("\n" + "=" * 72)
        print("  实数基线 (标准 Transformer — Cross-Entropy)")
        print(f"  d_model={cfg['d_model_real']} (real), "
              f"blocks={cfg['num_blocks']}, heads={cfg['num_heads']}")
        print("=" * 72)

        model_r = RealByteMLM(
            vocab_size=256,
            d_model=cfg['d_model_real'],
            num_heads=cfg['num_heads'],
            num_blocks=cfg['num_blocks'],
        ).to(device)

        n_params_r = sum(p.numel() for p in model_r.parameters())
        print(f"  参数量: {n_params_r:,} (实数)")

        out_r = os.path.join(args.output_dir, 'real')
        results['real'] = train_real(
            model_r, gen, cfg, device, out_r)

    # ---- 对比总结 ----
    if len(results) >= 2:
        print("\n" + "=" * 72)
        print("  对照总结")
        print("=" * 72)
        for name, r in results.items():
            print(f"  [{name:>8s}] Test Acc: {r['final_test_acc']:.2%} | "
                  f"Params: {r['total_params']:,} | "
                  f"Time: {r['training_time_sec']:.0f}s")
            if 'final_test_ppl' in r:
                print(f"  {'':>10s} PPL: {r['final_test_ppl']:.1f}")

    print(f"\n结果已保存到: {args.output_dir}/")
    print()


if __name__ == '__main__':
    main()
