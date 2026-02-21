import os
import random
import numpy as np
import torch
from datetime import datetime

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def abspath(p: str) -> str:
    return os.path.abspath(p)

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")