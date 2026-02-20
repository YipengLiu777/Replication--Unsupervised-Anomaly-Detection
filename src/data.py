import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def load_pkl(data_pkl: str, dirty_key: str, n_clean: int, n_dirty: int):
    with open(data_pkl, "rb") as f:
        obj = pickle.load(f)
    assert "clean" in obj and dirty_key in obj, f"keys={list(obj.keys())}"
    x_clean = obj["clean"][:n_clean]
    x_dirty = obj[dirty_key][:n_dirty] if n_dirty > 0 else obj[dirty_key][:0]
    return x_clean, x_dirty

class ImageDataset(Dataset):
    """
    If include_dirty=False => only clean samples used for training.
    If include_dirty=True  => concat clean + dirty for evaluation.
    """
    def __init__(self, x_clean: np.ndarray, x_dirty: np.ndarray, out_size: int, include_dirty: bool):
        assert x_clean.ndim == 4 and x_clean.shape[-1] == 3
        self.out_size = out_size
        if include_dirty and (x_dirty is not None) and (len(x_dirty) > 0):
            self.x = np.concatenate([x_clean, x_dirty], axis=0)
        else:
            self.x = x_clean

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]  # uint8 HWC
        if self.out_size != img.shape[0]:
            img = cv2.resize(img, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        t = torch.from_numpy(img)
        return t, t