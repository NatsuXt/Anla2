import torch
import torch.nn as nn
import sys
import os

# -------------------------------------------------------------------------
# [Path Fix] 文件位置: Anla/experiments/torus/train_torus_2d.py
# -------------------------------------------------------------------------
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.layers.embedding import ComplexEmbedding
from Anla.layers.positional import ComplexRotaryEmbedding
from Anla.layers.transformer_block import ComplexTransformerBlock
from Anla.layers.linear import ComplexLinear
from Anla.utils.torus_data import TorusDataGenerator

# Configuration [UPDATED]
CONFIG = {
    'grid_size': 8,
    'd_model': 128,          
    'num_heads': 8,          
    'seq_steps': 16,
    'batch_size': 32,
    'lr': 0.002,
    'weight_decay': 1e-4,
    'epochs': 10000,
    'mask_prob': 0.3,       # 单点遮挡概率
    'blind_span_prob': 0.4, # [NEW] 40% 的概率进行盲航训练
    'save_dir': 'checkpoints_2d'
}

# Derived Config
NUM_LOCS = CONFIG['grid_size'] ** 2
NUM_ACTIONS = 4
MASK_ID = NUM_LOCS + NUM_ACTIONS 
VOCAB_SIZE = MASK_ID + 1         

class AnlaTorusNavigator(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = ComplexEmbedding(vocab_size, d_model)
        self.rotary = ComplexRotaryEmbedding(d_model, max_seq_len=128)
        self.block = ComplexTransformerBlock(d_model, num_heads=CONFIG['num_heads'])
        self.head = ComplexLinear(d_model, vocab_size, bias=False) 

    def forward(self, x):
        z = self.embedding.forward(x)
        z = self.rotary.forward(z)
        z_out = self.block.forward(z, mask=None) 
        logits = self.head.forward(z_out)
        return torch.abs(logits)

    def manual_backward(self, grad_logits, lr):
        with torch.no_grad():
            z_transformer_out = self.head.input_cache
            complex_logits = nn.functional.linear(z_transformer_out, self.head.weight, self.head.bias)
            phase = torch.angle(complex_logits)
        
        grad_complex_logits = torch.polar(grad_logits, phase)
        grad_block_out = self.head.manual_backward(grad_complex_logits, lr, CONFIG['weight_decay'])
        grad_rotary_out = self.block.manual_backward(grad_block_out, lr, CONFIG['weight_decay'])
        grad_embed_out = self.rotary.manual_backward(grad_rotary_out)
        self.embedding.manual_backward(grad_embed_out, lr, CONFIG['weight_decay'])

def save_checkpoint(model, config, filename):
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    path = os.path.join(config['save_dir'], filename)
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, path)

def train_torus_navigation():
    print(f"=== Anla Task A: 2D Torus Navigation (Blind Span Mode) ===")
    print(f"Config: {CONFIG}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = TorusDataGenerator(CONFIG['grid_size'], CONFIG['seq_steps'], MASK_ID)
    model = AnlaTorusNavigator(VOCAB_SIZE, CONFIG['d_model']).to(device)
    
    # Optional: Load previous best model to fine-tune
    # If starting from scratch is preferred, comment this out.
    # model_path = os.path.join(CONFIG['save_dir'], 'best_torus_model.pth')
    # if os.path.exists(model_path):
    #     print(f"Loading previous weights from {model_path}...")
    #     ckpt = torch.load(model_path)
    #     model.load_state_dict(ckpt['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # [UPDATED] Pass blind_span_prob
        input_ids, target_ids = generator.generate_batch(CONFIG['batch_size'], 
                                                         mask_prob=CONFIG['mask_prob'],
                                                         blind_span_prob=CONFIG['blind_span_prob'])
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        logits_mag = model.forward(input_ids)
        loss = criterion(logits_mag.view(-1, VOCAB_SIZE), target_ids.view(-1))
        
        with torch.no_grad():
            probs = torch.softmax(logits_mag, dim=-1)
            grad_logits = probs.clone()
            flat_targets = target_ids.view(-1)
            flat_grad = grad_logits.view(-1, VOCAB_SIZE)
            mask = flat_targets != -100
            valid_targets = flat_targets[mask]
            
            row_indices = torch.arange(flat_grad.shape[0], device=device)[mask]
            flat_grad[row_indices, valid_targets] -= 1.0
            flat_grad[~mask, :] = 0.0
            num_valid = mask.sum()
            grad_logits = flat_grad.view_as(logits_mag) / num_valid if num_valid > 0 else torch.zeros_like(logits_mag)

        model.manual_backward(grad_logits, CONFIG['lr'])
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")
            
            # Debug Logic
            valid_mask = target_ids[0] != -100
            if valid_mask.any():
                idx = torch.where(valid_mask)[0][0].item()
                ctx_s = max(0, idx - 2)
                ctx_e = min(33, idx + 3)
                context_ids = input_ids[0, ctx_s:ctx_e].tolist()
                decoded_ctx = [generator.decode_token(t) for t in context_ids]
                
                true_id = target_ids[0, idx].item()
                pred_id = torch.argmax(logits_mag[0, idx]).item()
                
                true_str = generator.decode_token(true_id)
                pred_str = generator.decode_token(pred_id)
                
                status = "CORRECT" if true_id == pred_id else "FAIL"
                print(f"   [Debug] ...{' -> '.join(decoded_ctx)}...")
                print(f"           Expect: {true_str} | Pred: {pred_str} ({status})")

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(model, CONFIG, 'best_torus_model.pth')

    save_checkpoint(model, CONFIG, 'final_torus_model.pth')
    print("Training Complete.")

if __name__ == "__main__":
    train_torus_navigation()
