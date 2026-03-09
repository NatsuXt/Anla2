"""
保存位置: Anla/experiments/nlp/byte_mlm_v7.py

Byte-Level MLM — Autograd 复数 Transformer vs 实数 Transformer
=========================================================================

v7.3 — PyTorch Autograd 复数架构:
    所有复数层使用 PyTorch 原生 autograd (Wirtinger calculus),
    替代 Anla 的手动反向传播 + Polar Adam。

    保留的复数前向计算:
        · 复数嵌入 (ComplexEmbedding)
        · 复数 RoPE 位置编码
        · 复数 RMSNorm
        · 复数多头注意力 (Hermitian 内积)
        · 全纯激活函数 f(z) = z + α·z²
        · Weight Tying: logits = Re(z @ conj(W).T)

    替换的部分:
        · 手动 Wirtinger 反向传播 → PyTorch autograd
        · Polar Adam → AdamW (PyTorch 原生, 支持复数参数)
        · Boltzmann N 体力场 → Cross-Entropy on cosine logits

    目的:
        诊断 42.7% 天花板的根因:
        是手动反向传播的梯度累积误差, 还是复数架构本身?

用法:
    python -m Anla.experiments.nlp.byte_mlm_v7 --model complex
    python -m Anla.experiments.nlp.byte_mlm_v7 --model real
    python -m Anla.experiments.nlp.byte_mlm_v7 --model both
    python -m Anla.experiments.nlp.byte_mlm_v7 --model both --data-path Anla/data/tiny_shakespeare.txt
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)


# =====================================================================
#  数据
# =====================================================================

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tiny_shakespeare.txt")
    if os.path.exists(path):
        print(f"  [数据] 使用缓存: {path}")
        return path
    print(f"  [数据] 下载 tiny_shakespeare...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    print(f"  [数据] 完成: {os.path.getsize(path):,} bytes -> {path}")
    return path


class TextByteGenerator:
    def __init__(self, data_path, seq_len, mask_id, test_frac=0.1, seed=42):
        self.seq_len = seq_len
        self.mask_id = mask_id
        with open(data_path, 'rb') as f:
            raw = f.read()
        self.data = np.frombuffer(raw, dtype=np.uint8)
        split = int(len(self.data) * (1.0 - test_frac))
        self.train_data = self.data[:split]
        self.test_data = self.data[split:]
        print(f"  [数据] 总: {len(self.data):,}, "
              f"训练: {len(self.train_data):,}, 测试: {len(self.test_data):,}, "
              f"唯一 byte: {len(np.unique(self.data))}/256")

    def _generate_batch(self, data, batch_size, mask_mode='bert',
                        mask_prob=0.15, max_span=5):
        inp_list, tgt_list = [], []
        max_start = len(data) - self.seq_len
        for _ in range(batch_size):
            start = random.randint(0, max_start)
            seq = torch.tensor(data[start:start+self.seq_len].copy(), dtype=torch.long)
            inp, tgt = seq.clone(), torch.full_like(seq, -100)
            if mask_mode == 'bert':
                mask = torch.rand(self.seq_len) < mask_prob
                if not mask.any():
                    mask[random.randint(0, self.seq_len-1)] = True
                inp[mask] = self.mask_id
                tgt[mask] = seq[mask]
            elif mask_mode == 'span':
                ml = random.randint(1, max_span)
                ms = random.randint(0, self.seq_len - ml)
                inp[ms:ms+ml] = self.mask_id
                tgt[ms:ms+ml] = seq[ms:ms+ml]
            inp_list.append(inp)
            tgt_list.append(tgt)
        return torch.stack(inp_list), torch.stack(tgt_list)

    def generate_train_batch(self, batch_size, **kw):
        return self._generate_batch(self.train_data, batch_size, **kw)

    def generate_test_batch(self, batch_size, **kw):
        return self._generate_batch(self.test_data, batch_size, **kw)


# =====================================================================
#  Autograd 复数 Transformer 组件
# =====================================================================

def complex_kaiming_init_(tensor):
    """Kaiming 初始化复数张量。"""
    fan_in = tensor.shape[-1]
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        real = torch.randn_like(tensor.real) * std
        imag = torch.randn_like(tensor.imag) * std
        tensor.copy_(torch.complex(real, imag))
    return tensor


class AGComplexEmbedding(nn.Module):
    """复数嵌入层 (autograd)。初始化在单位圆附近。"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        w = torch.randn(num_embeddings, embedding_dim, dtype=torch.cfloat)
        w = w / (w.abs() + 1e-9)  # 归一化到单位模长
        self.weight = nn.Parameter(w)

    def forward(self, x):
        return self.weight[x]


class AGComplexRMSNorm(nn.Module):
    """复数 RMSNorm (autograd)。"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, z):
        # RMS of complex modulus
        rms = torch.sqrt(torch.mean(z.abs() ** 2, dim=-1, keepdim=True) + self.eps)
        return (z / rms) * self.scale


class AGComplexRotary(nn.Module):
    """复数 RoPE (autograd)。对复数向量施加旋转。"""
    def __init__(self, d_model, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, z):
        # z: (B, S, D) complex
        seq_len = z.shape[1]
        t = torch.arange(seq_len, device=z.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq)  # (S, D)
        rotors = torch.polar(torch.ones_like(angles), angles)  # e^{i·angle}
        return z * rotors.unsqueeze(0)  # (B, S, D) * (1, S, D)


class AGHolomorphicActivation(nn.Module):
    """全纯激活: f(z) = z + α·z² (autograd)。"""
    def __init__(self, d_model):
        super().__init__()
        # α: 逐通道复数参数
        alpha = torch.randn(d_model, dtype=torch.cfloat) * 0.01
        self.alpha = nn.Parameter(alpha)

    def forward(self, z):
        return z + self.alpha * z * z


class AGComplexAttention(nn.Module):
    """
    复数多头注意力 (autograd)。
    Hermitian 内积: score = Re(Q^H K) / sqrt(d_head)
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V 投影
        self.wq = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wk = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wv = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))
        self.wo = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, d_model, dtype=torch.cfloat)))

    def forward(self, x, mask=None):
        B, S, D = x.shape
        H, Dh = self.num_heads, self.d_head

        Q = F.linear(x, self.wq).view(B, S, H, Dh).transpose(1, 2)  # (B,H,S,Dh)
        K = F.linear(x, self.wk).view(B, S, H, Dh).transpose(1, 2)
        V = F.linear(x, self.wv).view(B, S, H, Dh).transpose(1, 2)

        # Hermitian 内积: Re(Q^H K) = Re(conj(Q) · K)
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.real(torch.matmul(Q.conj(), K.transpose(-2, -1))) * scale  # (B,H,S,S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # real softmax

        # attn (real) × V (complex)
        out = torch.matmul(attn.to(V.dtype), V)  # (B,H,S,Dh)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = F.linear(out, self.wo)
        return out


class AGComplexTransformerBlock(nn.Module):
    """Pre-Norm 复数 Transformer Block (autograd)。"""
    def __init__(self, d_model, num_heads=4, ff_mult=4):
        super().__init__()
        self.norm1 = AGComplexRMSNorm(d_model)
        self.attn = AGComplexAttention(d_model, num_heads)
        self.norm2 = AGComplexRMSNorm(d_model)
        ff_dim = d_model * ff_mult
        self.ff1 = nn.Parameter(complex_kaiming_init_(
            torch.empty(ff_dim, d_model, dtype=torch.cfloat)))
        self.act = AGHolomorphicActivation(ff_dim)
        self.ff2 = nn.Parameter(complex_kaiming_init_(
            torch.empty(d_model, ff_dim, dtype=torch.cfloat)))

    def forward(self, x, mask=None):
        # Attention sub-layer
        h = self.norm1(x)
        h = self.attn(h, mask)
        x = x + h

        # FFN sub-layer
        h = self.norm2(x)
        h = F.linear(h, self.ff1)
        h = self.act(h)
        h = F.linear(h, self.ff2)
        x = x + h

        return x


class ComplexByteMLM(nn.Module):
    """
    Autograd 复数 Transformer — Byte-level MLM。

    前向计算与 Anla v7 等价:
        ComplexEmbedding → ComplexRotary → [ComplexBlock × N] → cosine logits
    梯度: PyTorch autograd (精确 Wirtinger calculus)
    优化: AdamW (标准, 支持复数参数)
    """
    def __init__(self, vocab_size=256, d_model=64, num_heads=4, num_blocks=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = AGComplexEmbedding(vocab_size + 1, d_model)
        self.rotary = AGComplexRotary(d_model, max_seq_len=512)
        self.blocks = nn.ModuleList([
            AGComplexTransformerBlock(d_model, num_heads, ff_mult=4)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Returns:
            logits: (B, S, V) real — cosine 相似度作为 logits
        """
        z = self.embedding(x)
        z = self.rotary(z)
        for block in self.blocks:
            z = block(z)

        # Weight Tying: cosine logits
        W = self.embedding.weight[:self.vocab_size]    # (V, D) complex
        z_norm = z.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
        w_norm = W.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
        z_hat = z / z_norm
        w_hat = W / w_norm

        # Re(ẑ^H ŵ_k) = cos_C(z, w_k)
        logits = torch.real(z_hat @ w_hat.conj().T)    # (B, S, V)

        # 温度缩放: cosine ∈ [-1,1], 除以 τ 使 softmax 有足够区分度
        # 类似 CLIP 的 learnable temperature
        logits = logits / 0.07  # τ = 0.07, CLIP 默认值

        return logits


# =====================================================================
#  实数基线 (完全不变)
# =====================================================================

class RealRotaryEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, base=10000.0):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        cos_f, sin_f = freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.zeros_like(x)
        out[..., ::2] = x1 * cos_f - x2 * sin_f
        out[..., 1::2] = x1 * sin_f + x2 * cos_f
        return out


class RealByteMLM(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, num_heads=4,
                 num_blocks=3, ff_mult=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.rotary = RealRotaryEmbedding(d_model, max_seq_len=512)
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
                    nn.Linear(d_model * ff_mult, d_model)),
            }))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embedding(x)
        h = self.rotary(h)
        for block in self.blocks:
            r = h; h = block['norm1'](h)
            h, _ = block['attn'](h, h, h, need_weights=False); h = r + h
            r = h; h = block['norm2'](h); h = block['ff'](h); h = r + h
        h = self.output_norm(h)
        return F.linear(h, self.embedding.weight[:self.vocab_size])


# =====================================================================
#  评估
# =====================================================================

@torch.no_grad()
def evaluate_model(model, gen, batch_size, device, num_batches=10,
                   mask_prob=0.15):
    """统一评估: argmax logits + CE perplexity。"""
    model.eval()
    total_correct, total_count, total_loss = 0, 0, 0.0
    for _ in range(num_batches):
        inp, tgt = gen.generate_test_batch(batch_size, mask_prob=mask_prob)
        inp, tgt = inp.to(device), tgt.to(device)
        valid = (tgt != -100)
        if not valid.any():
            continue
        logits = model(inp)
        lv, tv = logits[valid], tgt[valid]
        total_loss += F.cross_entropy(lv, tv, reduction='sum').item()
        total_correct += (lv.argmax(-1) == tv).sum().item()
        total_count += tv.shape[0]
    acc = total_correct / max(total_count, 1)
    ppl = math.exp(min(total_loss / max(total_count, 1), 20.0))
    return acc, ppl


# =====================================================================
#  统一训练函数 (autograd, 两种模型共用)
# =====================================================================

def train_model(model, gen, cfg, device, output_dir, model_name='complex'):
    os.makedirs(output_dir, exist_ok=True)

    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr_base = cfg['lr']
    warmup = cfg['warmup_epochs']
    wd = cfg['weight_decay']
    mask_prob = cfg['mask_prob']
    log_interval = cfg['log_interval']

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base,
                                  weight_decay=wd, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, max(1, s) / warmup))

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

        logits = model(inp)

        if valid.any():
            loss = F.cross_entropy(logits[valid], tgt[valid])
        else:
            loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % log_interval == 0:
            with torch.no_grad():
                tr_acc = (logits[valid].argmax(-1) == tgt[valid]).float().mean().item() \
                    if valid.any() else 0.0

            test_acc, test_ppl = -1.0, -1.0
            if epoch % (log_interval * 5) == 0 or epoch == epochs - 1:
                test_acc, test_ppl = evaluate_model(
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
            print(f"  [{model_name}] Ep {epoch:05d} | "
                  f"L: {loss.item():.4f} | Tr: {tr_acc:.2%}{test_str}{ppl_str}")

    elapsed = time.time() - t_start
    final_acc, final_ppl = evaluate_model(
        model, gen, batch_size, device, num_batches=20)

    result = {
        'model': model_name,
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

    print(f"\n  [{model_name}] 完成: Train {best_train_acc:.2%}, "
          f"Test {final_acc:.2%}, PPL {final_ppl:.1f}, {elapsed:.0f}s")
    return result


# =====================================================================
#  主入口
# =====================================================================

DEFAULT_CFG = {
    'd_model_complex': 64,
    'd_model_real': 128,
    'num_heads': 4,
    'num_blocks': 3,
    'seq_len': 64,
    'batch_size': 64,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 15000,
    'warmup_epochs': 500,
    'mask_prob': 0.15,
    'log_interval': 200,
}


def main():
    parser = argparse.ArgumentParser(
        description="Byte-level MLM: Autograd 复数 vs 实数 Transformer")
    parser.add_argument('--model', type=str, default='complex',
                        choices=['complex', 'real', 'both'])
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_ANLA_ROOT, 'Logs', 'nlp_byte_mlm_v7'))
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    cfg = DEFAULT_CFG.copy()
    if args.epochs: cfg['epochs'] = args.epochs
    if args.seq_len: cfg['seq_len'] = args.seq_len
    if args.batch_size: cfg['batch_size'] = args.batch_size

    if args.data_path is None:
        data_path = download_tiny_shakespeare(os.path.join(_ANLA_ROOT, 'data'))
    else:
        data_path = args.data_path

    gen = TextByteGenerator(data_path, cfg['seq_len'], mask_id=256, test_frac=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    if args.model in ('complex', 'both'):
        print("\n" + "=" * 72)
        print("  复数模型 (Autograd — 复数 Transformer + Cosine + CE)")
        d = cfg['d_model_complex']
        print(f"  d_model={d} (complex), blocks={cfg['num_blocks']}, "
              f"heads={cfg['num_heads']}")
        print("=" * 72)

        model_c = ComplexByteMLM(
            vocab_size=256, d_model=d,
            num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
        ).to(device)

        n_p = sum(p.numel() for p in model_c.parameters())
        n_r = sum(p.numel() * (2 if p.is_complex() else 1)
                  for p in model_c.parameters())
        print(f"  参数量: {n_p:,} (复数), 实数等效: {n_r:,}")

        results['complex'] = train_model(
            model_c, gen, cfg, device,
            os.path.join(args.output_dir, 'complex'), 'complex')

    if args.model in ('real', 'both'):
        print("\n" + "=" * 72)
        print("  实数基线 (标准 Transformer — CE + AdamW)")
        d = cfg['d_model_real']
        print(f"  d_model={d} (real), blocks={cfg['num_blocks']}, "
              f"heads={cfg['num_heads']}")
        print("=" * 72)

        model_r = RealByteMLM(
            vocab_size=256, d_model=d,
            num_heads=cfg['num_heads'], num_blocks=cfg['num_blocks'],
        ).to(device)
        print(f"  参数量: {sum(p.numel() for p in model_r.parameters()):,}")

        results['real'] = train_model(
            model_r, gen, cfg, device,
            os.path.join(args.output_dir, 'real'), 'real')

    if len(results) >= 2:
        print("\n" + "=" * 72)
        print("  对照总结")
        print("=" * 72)
        for name, r in results.items():
            ppl = f", PPL: {r['final_test_ppl']:.1f}" \
                if 'final_test_ppl' in r else ""
            print(f"  [{name:>8s}] Test: {r['final_test_acc']:.2%}{ppl}"
                  f" | Params: {r['total_params']:,}"
                  f" | Time: {r['training_time_sec']:.0f}s")

    print(f"\n结果已保存到: {args.output_dir}/")


if __name__ == '__main__':
    main()
