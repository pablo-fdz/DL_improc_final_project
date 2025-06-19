import random
import torch
import numpy as np

def set_seed(seed=42):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For Python's built-in random module (if used)
    random.seed(seed)
    print(f"Seed set to {seed} for NumPy, Torch and Random for reproducibility.")