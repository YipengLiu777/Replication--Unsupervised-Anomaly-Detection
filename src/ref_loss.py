import numpy as np
import torch

@torch.no_grad()
def build_href_from_clean(
    model,
    dataset,
    n_clean: int,
    M: int = 1024,
    batch_ref: int = 128,
    device: torch.device = torch.device("cpu"),
):
    """
    Build reference latent set H_ref using ONLY clean samples.
    Assumption: dataset indices [0, n_clean) are clean.
    Returns: H_ref on `device` in FP32
    """
    model.eval()
    n_clean = int(n_clean)
    M = int(M)
    batch_ref = int(batch_ref)

    if M > n_clean:
        M = n_clean

    idx = np.random.choice(n_clean, size=M, replace=False)

    z_list = []
    for s in range(0, M, batch_ref):
        sub = idx[s : s + batch_ref]
        xb = torch.stack([dataset[i][0] for i in sub], dim=0).to(device, non_blocking=True)

        # force FP32 latent for stability
        with torch.cuda.amp.autocast(enabled=False):
            _, z = model(xb)
            z_list.append(z.float().detach().cpu())

    H_ref_cpu = torch.cat(z_list, dim=0)  # CPU FP32
    H_ref = H_ref_cpu.to(device, non_blocking=True)  # move to device
    return H_ref


def latent_ref_loss_knn(z: torch.Tensor, H_ref: torch.Tensor, knn_k: int = 1) -> torch.Tensor:
    """
    Paper-like reference loss (nearest neighbor):
        Lref = mean_i min_j ||z_i - h_j||^2

    For stability, we also support kNN-mean (knn_k>1):
        Lref = mean_i mean_{j in kNN(i)} ||z_i - h_j||^2

    Inputs assumed on same device. Computed in FP32.
    """
    z = z.float()
    H_ref = H_ref.float()

    # dist2 = ||z||^2 + ||h||^2 - 2 z h^T
    z2 = (z * z).sum(dim=1, keepdim=True)          # (B,1)
    h2 = (H_ref * H_ref).sum(dim=1).unsqueeze(0)   # (1,M)
    dist2 = z2 + h2 - 2.0 * (z @ H_ref.t())        # (B,M)
    dist2 = torch.clamp(dist2, min=0.0)

    k = int(knn_k)
    k = max(1, k)
    k = min(k, dist2.shape[1])

    if k == 1:
        # nearest neighbor (paper)
        mins, _ = dist2.min(dim=1)
        return mins.mean()
    else:
        # kNN mean (more stable)
        vals, _ = torch.topk(dist2, k=k, dim=1, largest=False, sorted=False)  # (B,k)
        return vals.mean()