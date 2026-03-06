import torch
import random

class TorusDataGenerator:
    """
    [Anla 2D Navigation Testbed - Advanced]
    生成二维环面(Torus)上的导航序列。
    
    [Upgrade]: 支持 Blind Span Masking (盲航训练)，
    强迫模型在缺乏中间坐标路标的情况下，进行长程路径积分。
    """
    def __init__(self, grid_size=8, seq_steps=16, mask_token_id=None):
        self.grid_size = grid_size
        self.num_locs = grid_size * grid_size
        self.num_actions = 4
        self.action_offset = self.num_locs
        self.mask_id = mask_token_id if mask_token_id is not None else (self.num_locs + self.num_actions)
        self.seq_steps = seq_steps 
        self.full_seq_len = seq_steps * 2 + 1

    def _move(self, x, y, action):
        if action == 0:   y = (y - 1) % self.grid_size # UP
        elif action == 1: y = (y + 1) % self.grid_size # DOWN
        elif action == 2: x = (x - 1) % self.grid_size # LEFT
        elif action == 3: x = (x + 1) % self.grid_size # RIGHT
        return x, y

    def generate_batch(self, batch_size, mask_prob=0.3, blind_span_prob=0.3):
        """
        Args:
            mask_prob: 随机单点遮挡概率 (用于基础完形填空)
            blind_span_prob: 盲航片段概率 (用于路径积分)
        """
        input_ids = []
        target_ids = []
        
        for _ in range(batch_size):
            # 1. Generate Trajectory
            start_x = random.randint(0, self.grid_size - 1)
            start_y = random.randint(0, self.grid_size - 1)
            curr_x, curr_y = start_x, start_y
            
            seq = []
            start_token = curr_y * self.grid_size + curr_x
            seq.append(start_token) # Pos 0
            
            # Record types for masking logic: 0=Pos, 1=Act
            token_types = [0] 
            
            for _ in range(self.seq_steps):
                action = random.randint(0, 3)
                action_token = self.action_offset + action
                
                next_x, next_y = self._move(curr_x, curr_y, action)
                next_loc_token = next_y * self.grid_size + next_x
                
                seq.append(action_token) # Pos 2i+1
                token_types.append(1)
                
                seq.append(next_loc_token) # Pos 2i+2
                token_types.append(0)
                
                curr_x, curr_y = next_x, next_y
                
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            
            # 2. Advanced Masking
            inp = seq_tensor.clone()
            tgt = torch.full_like(seq_tensor, -100)
            
            # 策略 A: 盲航模式 (Blind Span)
            # 随机选择一段区间，把里面的所有 坐标(Type 0) 都 Mask 掉，保留 动作(Type 1)
            if random.random() < blind_span_prob:
                # 随机长度 2 到 5
                span_len = random.randint(2, 5)
                # 随机起点 (保证是坐标位置)
                # 序列结构: P0, A1, P1, A2, P2...
                # valid start indices for Pos: 0, 2, 4...
                max_step_idx = self.seq_steps - span_len
                if max_step_idx > 0:
                    start_step = random.randint(0, max_step_idx)
                    # Masking loop
                    for k in range(span_len):
                        # P_start, A, P_next, A...
                        # Masking the coordinates in the chain
                        # The coordinate at step i is at index 2*i (approx)
                        # indices: 0, 2, 4...
                        
                        # Current Pos Index
                        idx = (start_step + k + 1) * 2 
                        if idx < len(seq):
                            inp[idx] = self.mask_id
                            tgt[idx] = seq_tensor[idx]
            
            # 策略 B: 随机噪声 (Random Noise) - 补充策略
            else:
                for i in range(1, len(seq)):
                    if random.random() < mask_prob:
                        inp[i] = self.mask_id
                        tgt[i] = seq_tensor[i]
            
            input_ids.append(inp)
            target_ids.append(tgt)
            
        return torch.stack(input_ids), torch.stack(target_ids)

    def decode_token(self, token_id):
        if token_id < self.num_locs:
            y = token_id // self.grid_size
            x = token_id % self.grid_size
            return f"({x},{y})"
        elif token_id < self.action_offset + 4:
            acts = ["UP", "DN", "LF", "RT"]
            return acts[token_id - self.action_offset]
        elif token_id == self.mask_id:
            return "[?]"
        else:
            return "UNK"
