import numpy as np
import torch

@torch.no_grad()
def compute_pcc_mse(model, dataloader, device):
    model.eval()
    pcc_all, mse_all = [], []
    for xb, _ in dataloader:
        xb = xb.to(device, non_blocking=True)
        xrec, _ = model(xb)

        mse = ((xrec - xb) ** 2).flatten(1).mean(dim=1)

        x = xb.flatten(1)
        y = xrec.flatten(1)
        xm = x - x.mean(dim=1, keepdim=True)
        ym = y - y.mean(dim=1, keepdim=True)
        num = (xm * ym).sum(dim=1)
        den = torch.sqrt((xm * xm).sum(dim=1) * (ym * ym).sum(dim=1) + 1e-12)
        pcc = num / den

        pcc_all.append(pcc.detach().cpu())
        mse_all.append(mse.detach().cpu())

    pcc = torch.cat(pcc_all).numpy().astype(np.float64)
    mse = torch.cat(mse_all).numpy().astype(np.float64)
    return pcc, mse

def median_filter_1d(x: np.ndarray, k: int = 101):
    assert k % 2 == 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        out[i] = np.median(xpad[i:i+k])
    return out