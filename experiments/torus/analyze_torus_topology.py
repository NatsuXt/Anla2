import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# [Path Fix] 文件位置: Anla/experiments/torus/analyze_torus_topology.py
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ANLA_ROOT = os.path.abspath(os.path.join(_FILE_DIR, '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ANLA_ROOT, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from Anla.experiments.torus.train_torus_2d import AnlaTorusNavigator, CONFIG, NUM_LOCS

def load_model(device):
    model_path = os.path.join('checkpoints_2d', 'final_torus_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join('checkpoints_2d', 'best_torus_model.pth')
        
    print(f">> Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    saved_config = checkpoint['config']
    
    # [FIX]: Re-calculate vocab_size from grid_size
    # Logic from train_torus_2d.py:
    # NUM_LOCS = grid_size ** 2
    # NUM_ACTIONS = 4
    # MASK_ID = NUM_LOCS + NUM_ACTIONS
    # VOCAB_SIZE = MASK_ID + 1
    
    grid_size = saved_config['grid_size']
    num_locs = grid_size ** 2
    vocab_size = num_locs + 4 + 1 
    
    model = AnlaTorusNavigator(vocab_size, saved_config['d_model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def analyze_topology():
    print("=== Anla 2D Topology Analysis ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = load_model(device)
    except FileNotFoundError:
        print("[Error] Model checkpoint not found. Run train_torus_2d.py first.")
        return

    # 1. Extract Location Embeddings
    # 只取前 NUM_LOCS 个 Token (Coordinates)
    # Re-calculate NUM_LOCS just in case (though imported)
    grid_size = CONFIG['grid_size']
    num_locs = grid_size ** 2
    
    loc_embeds = model.embedding.weight.data[:num_locs, :]
    
    # 2. Convert to Phase (Angle)
    phases = torch.angle(loc_embeds).cpu().numpy()
    
    d_model = loc_embeds.shape[1] # Use actual model dimension
    
    # Reshape to (8, 8, D_Model)
    grid_phases = phases.reshape(grid_size, grid_size, d_model)
    
    print(f">> Analyzing {d_model} dimensions for topological structure...")
    
    # 3. Find Best X-Axis and Y-Axis Dimensions
    x_scores = []
    y_scores = []
    
    for dim in range(d_model):
        p_map = grid_phases[:, :, dim] # 8x8
        
        # Metric: Consistency along axes
        # Y-Invariance (Good for X-encoding): mean length of column vectors
        col_consistency = np.mean(np.abs(np.mean(np.exp(1j * p_map), axis=0)))
        
        # X-Invariance (Good for Y-encoding): mean length of row vectors
        row_consistency = np.mean(np.abs(np.mean(np.exp(1j * p_map), axis=1)))
        
        x_scores.append(col_consistency) # High if Y is constant -> Encodes X
        y_scores.append(row_consistency) # High if X is constant -> Encodes Y

    best_x_dim = np.argmax(x_scores)
    best_y_dim = np.argmax(y_scores)
    
    print(f">> Best X-Encoding Dimension: {best_x_dim} (Score: {x_scores[best_x_dim]:.4f})")
    print(f">> Best Y-Encoding Dimension: {best_y_dim} (Score: {y_scores[best_y_dim]:.4f})")
    
    # 4. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot X-Encoding Dim
    im1 = axes[0].imshow(grid_phases[:, :, best_x_dim], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(f"Dim {best_x_dim}: X-Axis Topology\n(Target: Vertical Stripes)")
    axes[0].set_xlabel("X Index")
    axes[0].set_ylabel("Y Index")
    fig.colorbar(im1, ax=axes[0], label="Phase")
    
    # Plot Y-Encoding Dim
    im2 = axes[1].imshow(grid_phases[:, :, best_y_dim], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Dim {best_y_dim}: Y-Axis Topology\n(Target: Horizontal Stripes)")
    axes[1].set_xlabel("X Index")
    axes[1].set_ylabel("Y Index")
    fig.colorbar(im2, ax=axes[1], label="Phase")
    
    plt.suptitle("Anla 2D Manifold Phase Reconstruction")
    plt.tight_layout()
    
    save_path = "topology_vis.png"
    plt.savefig(save_path)
    print(f">> Visualization saved to {save_path}")
    print(">> Please check the image.")

if __name__ == "__main__":
    analyze_topology()
