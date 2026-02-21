# src/metrics.py
import numpy as np
import torch
from torch.utils.data import DataLoader

EPS = 1e-12

@torch.no_grad()
def compute_pcc_mse(model, dataset, batch_size: int, device, num_workers: int = 0):
    """
    Return:
      pcc: (N,) numpy float32   Pearson corr (per-image, flatten all pixels)
      mse: (N,) numpy float32   mean squared error (per-image)
    """
    model.eval()
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    pcc_all = []
    mse_all = []

    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        xrec, _ = model(xb)

        # per-sample MSE
        mse = ((xrec - xb) ** 2).flatten(1).mean(dim=1)

        # per-sample PCC
        x = xb.flatten(1)
        y = xrec.flatten(1)
        xm = x - x.mean(dim=1, keepdim=True)
        ym = y - y.mean(dim=1, keepdim=True)

        num = (xm * ym).sum(dim=1)
        den = torch.sqrt((xm * xm).sum(dim=1) * (ym * ym).sum(dim=1) + EPS)
        pcc = num / den

        pcc_all.append(pcc.detach().cpu())
        mse_all.append(mse.detach().cpu())

    pcc = torch.cat(pcc_all).numpy().astype(np.float32)
    mse = torch.cat(mse_all).numpy().astype(np.float32)
    return pcc, mse


def median_filter_1d(x: np.ndarray, k: int = 101):
    """
    Simple 1D median filter (edge-padded).
    Paper uses ~100; we use odd window by default.
    """
    x = np.asarray(x)
    assert k % 2 == 1, "median filter window k must be odd"
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(xpad[pad:-pad], dtype=np.float64)

    for i in range(len(out)):
        out[i] = np.median(xpad[i : i + k])
    return out.astype(np.float32)

@torch.no_grad()
def score_dataset(model, dataset, device, batch_size: int = 256, metric: str = "pcc", med_k: int = 101):
    """
    Used by train.py to build pseudo-clean set.
    Returns:
      score: (N,) np.float32
        - if metric == "pcc": score = PCC_med (higher is more normal)
        - if metric == "mse": score = -MSE (higher is more normal)
    """
    pcc, mse = compute_pcc_mse(model, dataset, batch_size=batch_size, device=device, num_workers=0)

    if metric.lower() == "pcc":
        pcc_med = median_filter_1d(pcc, k=med_k)
        return pcc_med.astype(np.float32)

    if metric.lower() == "mse":
        # lower mse => more normal, so use negative for "higher is better"
        return (-mse).astype(np.float32)

    raise ValueError(f"Unknown metric='{metric}'. Use 'pcc' or 'mse'.")