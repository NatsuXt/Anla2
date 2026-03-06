import torch
import torch.nn as nn
import sys
import os

# [Path Fix] 文件位置: Anla/tests/test_bidirectional_inference.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)
from Anla.experiments.ring.train_ring_masking import AnlaManifoldInpainter, CONFIG

def load_trained_model(device):
    model_path = os.path.join('checkpoints', 'final_ring_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join('checkpoints', 'best_ring_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Please run train_ring_masking.py first.")
        
    print(f">> Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    saved_config = checkpoint['config']
    
    model = AnlaManifoldInpainter(saved_config['vocab_size'], saved_config['d_model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, saved_config

def generate_valid_ring_sequence(start_val, length, vocab_size):
    """辅助函数：生成一个合法的圆环序列"""
    return [(start_val + i) % vocab_size for i in range(length)]

def test_bidirectional_capabilities():
    print("=== Anla Bidirectional Inference Test (Correct Distribution) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, config = load_trained_model(device)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    MASK = config['mask_token_id']
    SEQ_LEN = config['seq_len'] # 32
    VOCAB = config['vocab_size']
    
    # 我们构造完整的合法序列，然后只 mask 掉我们想测试的部分
    # 这样可以保证上下文充裕，符合训练分布
    
    # Case 1: [M, M, 5, 6, 7...] -> Expect 3, 4
    # 构造以 3 开头的序列: [3, 4, 5, 6...]
    seq_1 = generate_valid_ring_sequence(start_val=3, length=SEQ_LEN, vocab_size=VOCAB)
    input_1 = list(seq_1)
    input_1[0] = MASK
    input_1[1] = MASK
    
    # Case 2: [M, M, 0, 1, 2...] -> Expect 62, 63
    # 构造以 62 开头的序列: [62, 63, 0, 1...]
    seq_2 = generate_valid_ring_sequence(start_val=62, length=SEQ_LEN, vocab_size=VOCAB)
    input_2 = list(seq_2)
    input_2[0] = MASK
    input_2[1] = MASK

    # Case 3: [62, M, M, 1, 2...] -> Expect 63, 0
    # 构造以 62 开头的序列: [62, 63, 0, 1...]
    seq_3 = generate_valid_ring_sequence(start_val=62, length=SEQ_LEN, vocab_size=VOCAB)
    input_3 = list(seq_3)
    input_3[1] = MASK # Mask "63"
    input_3[2] = MASK # Mask "0"
    
    test_cases = [
        {
            "name": "Case 1: Left-Side Blind (Retro-diction)",
            "desc": "Input: [M, M, 5, 6...]. Expect: [3, 4].",
            "input_full": input_1,
            "check_indices": [0, 1],
            "expect": [3, 4]
        },
        {
            "name": "Case 2: Zero-Crossing Reverse (Topology Test)",
            "desc": "Input: [M, M, 0, 1...]. Expect: [62, 63].",
            "input_full": input_2, 
            "check_indices": [0, 1],
            "expect": [62, 63]
        },
        {
            "name": "Case 3: Island Repair (Bidirectional)",
            "desc": "Input: [62, M, M, 1...]. Expect: [63, 0].",
            "input_full": input_3, 
            "check_indices": [1, 2],
            "expect": [63, 0]
        }
    ]
    
    print("\n" + "="*100)
    print(f"{'TEST CASE':<40} | {'PREDICTED':<20} | {'EXPECTED':<10} | {'STATUS'}")
    print("="*100)
    
    total_passed = 0
    
    for case in test_cases:
        print(f"\n>> Running: {case['name']}")
        
        inp_tensor = torch.tensor([case['input_full']], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(inp_tensor)
            preds = torch.argmax(logits, dim=-1) # (1, 32)
        
        # Validation
        pred_vals = []
        check_indices = case['check_indices']
        for idx in check_indices:
            pred_vals.append(preds[0, idx].item())
            
        is_correct = (pred_vals == case['expect'])
        status = "PASS" if is_correct else "FAIL"
        if is_correct: total_passed += 1
        
        # Display
        # Format input: show first 6 tokens...
        disp_in = case['input_full'][:6]
        disp_in = ['M' if x == MASK else x for x in disp_in]
        
        print(f"   Input Head  : {disp_in}...")
        print(f"   Target Pred : {pred_vals}")
        print(f"   Expected    : {case['expect']}")
        print(f"   Result      : [{status}]")

    print("\n" + "="*100)
    print(f"Summary: {total_passed}/{len(test_cases)} Tests Passed.")

if __name__ == "__main__":
    test_bidirectional_capabilities()
