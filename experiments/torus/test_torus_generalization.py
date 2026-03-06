import torch
import sys
import os

# [Path Fix] 文件位置: Anla/experiments/torus/test_torus_generalization.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)
from Anla.experiments.torus.train_torus_2d import AnlaTorusNavigator, CONFIG, VOCAB_SIZE, NUM_LOCS, NUM_ACTIONS, MASK_ID

def load_model(device):
    model_path = os.path.join('checkpoints_2d', 'final_torus_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join('checkpoints_2d', 'best_torus_model.pth')
    print(f">> Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    saved_config = checkpoint['config']
    
    grid_size = saved_config['grid_size']
    num_locs = grid_size ** 2
    vocab_size = num_locs + 4 + 1
    
    model = AnlaTorusNavigator(vocab_size, saved_config['d_model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, saved_config

def get_loc_id(x, y, grid_size=8):
    return y * grid_size + x

def get_action_id(name, num_locs=64):
    mapping = {'UP':0, 'DN':1, 'LF':2, 'RT':3}
    return num_locs + mapping[name]

def decode(token_id, grid_size=8, num_locs=64):
    if token_id < num_locs:
        y = token_id // grid_size
        x = token_id % grid_size
        return f"({x},{y})"
    elif token_id < num_locs + 4:
        acts = ["UP", "DN", "LF", "RT"]
        return acts[token_id - num_locs]
    elif token_id == num_locs + 4:
        return "[?]"
    return "UNK"

def make_input(seq_list, seq_steps, device):
    # Construct padded input of full training length
    full_len = seq_steps * 2 + 1
    inp = [MASK_ID] * full_len
    for i, val in enumerate(seq_list):
        if i < full_len:
            inp[i] = val
    return torch.tensor([inp], dtype=torch.long).to(device)

def test_generalization():
    print("=== Anla 2D Generalization Stress Test (Corrected) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, config = load_model(device)
    except FileNotFoundError:
        print("[Error] No model found.")
        return
        
    GRID = config['grid_size'] # 8
    LOCS = GRID * GRID
    STEPS = config['seq_steps']
    
    # --- Case Definitions ---
    start_id = get_loc_id(4,4, GRID)
    act_rt = get_action_id('RT', LOCS)
    act_up = get_action_id('UP', LOCS)
    target_54 = get_loc_id(5,4, GRID)
    target_43 = get_loc_id(4,3, GRID)
    
    # 1. Basic Move
    input_1 = make_input([start_id, act_rt, MASK_ID], STEPS, device)
    expect_1 = target_54
    
    # 2. Action Retro
    input_2 = make_input([start_id, MASK_ID, target_43], STEPS, device)
    expect_2 = act_up
    
    # 4. Boundary
    p77 = get_loc_id(7,7, GRID)
    p07 = get_loc_id(0,7, GRID)
    input_4 = make_input([p77, act_rt, MASK_ID], STEPS, device)
    expect_4 = p07

    # One-Shot Tests
    tests = [
        ("Basic Move (Right)", input_1, 2, expect_1),
        ("Action Retro (Up)", input_2, 1, expect_2),
        ("Boundary Hopping", input_4, 2, expect_4)
    ]
    
    print(f"{'TEST NAME':<30} | {'PRED':<10} | {'EXPECT':<10} | {'STATUS'}")
    print("-" * 70)
    
    for name, inp, idx, exp in tests:
        with torch.no_grad():
            logits = model(inp)
            pred_id = torch.argmax(logits[0, idx]).item()
        
        pred_str = decode(pred_id, GRID, LOCS)
        exp_str = decode(exp, GRID, LOCS)
        status = "PASS" if pred_id == exp else "FAIL"
        print(f"{name:<30} | {pred_str:<10} | {exp_str:<10} | {status}")

    # --- Special Case 3: Blind Path Integration (Auto-regressive) ---
    print("\n>> Running Case 3: Blind Path Integration (Step-by-Step)")
    # Path: (0,0) -> RT -> [M] -> RT -> [M] -> UP -> [M] -> UP -> [M]
    # Expect sequence: (1,0) -> (2,0) -> (2,7) -> (2,6)
    
    p00 = get_loc_id(0,0, GRID)
    seq_so_far = [p00]
    actions = [act_rt, act_rt, act_up, act_up]
    expected_path = [
        get_loc_id(1,0, GRID),
        get_loc_id(2,0, GRID),
        get_loc_id(2,7, GRID), # UP is y-1
        get_loc_id(2,6, GRID)
    ]
    
    all_steps_passed = True
    
    print(f"   Start at (0,0). Following path: RT, RT, UP, UP")
    
    for i, action in enumerate(actions):
        # Construct input: Current History + Next Action + MASK
        # Seq: P0, A0, M, A1, M...
        # We need to reconstruct the interleaved sequence
        current_input_list = []
        # Rebuild P, A, P, A sequence from history
        for j in range(len(seq_so_far)):
            current_input_list.append(seq_so_far[j])
            if j < len(seq_so_far) - 1:
                # We don't track past actions in this simple loop, 
                # but we need them for input.
                # Let's fix this structure.
                pass
        
        # Easier way: Keep a growing list of tokens
        if i == 0:
            token_list = [p00, action, MASK_ID]
        else:
            # Append new action and mask
            token_list.append(action)
            token_list.append(MASK_ID)
            
        inp_tensor = make_input(token_list, STEPS, device)
        target_idx = len(token_list) - 1
        
        with torch.no_grad():
            logits = model(inp_tensor)
            pred_id = torch.argmax(logits[0, target_idx]).item()
            
        pred_str = decode(pred_id, GRID, LOCS)
        exp_id = expected_path[i]
        exp_str = decode(exp_id, GRID, LOCS)
        
        is_correct = (pred_id == exp_id)
        status = "OK" if is_correct else "ERR"
        print(f"   Step {i+1} (Action {decode(action, GRID, LOCS)}): Pred {pred_str} vs Exp {exp_str} -> {status}")
        
        if not is_correct:
            all_steps_passed = False
            # Self-Correction: Feed the CORRECT token to next step?
            # Or feed the PRED token?
            # Auto-regressive usually feeds PRED.
            # Let's feed PRED to see if it recovers or diverges.
            seq_so_far.append(pred_id)
            # Update token list for next iter
            token_list[-1] = pred_id # Replace MASK with Pred
        else:
            seq_so_far.append(pred_id)
            token_list[-1] = pred_id

    final_status = "PASS" if all_steps_passed else "FAIL"
    print(f"   Case 3 Result: {final_status}")

if __name__ == "__main__":
    test_generalization()
