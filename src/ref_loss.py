import numpy as np
import torch


@torch.no_grad()
def build_href_from_clean(model, dataset, n_clean: int, M: int, batch_ref: int, device):
    """
    (旧版/半监督用法) 仅从 [0, n_clean) 采样构建 reference latent set.
    """
    idx = np.arange(int(n_clean))
    return build_href_from_indices(model, dataset, idx, M=M, batch_ref=batch_ref, device=device)


@torch.no_grad()
def build_href_from_indices(model, dataset, indices, M: int, batch_ref: int, device):
    """
    (纯无监督/通用) 从给定 indices 里随机采样 M 个样本，encode 成 latent，组成 H_ref。

    indices: list/np.ndarray of candidate indices
    """
    model.eval()
    indices = np.asarray(indices, dtype=np.int64)
    assert indices.ndim == 1 and len(indices) > 0, "indices must be 1D non-empty"

    if len(indices) >= M:
        choice = np.random.choice(indices, size=M, replace=False)
    else:
        # 候选集不够时允许有放回采样
        choice = np.random.choice(indices, size=M, replace=True)

    z_list = []
    for s in range(0, M, batch_ref):
        sub = choice[s : s + batch_ref]
        xb = torch.stack([dataset[int(i)][0] for i in sub], dim=0).to(device, non_blocking=True)

        # FP32：稳定
        with torch.cuda.amp.autocast(enabled=False):
            _, z = model(xb)
            z_list.append(z.float().detach().cpu())

    H_ref = torch.cat(z_list, dim=0).to(device, non_blocking=True)
    return H_ref


def latent_ref_loss_knn(z: torch.Tensor, H_ref: torch.Tensor, knn_k: int = 1) -> torch.Tensor:
    """
    kNN reference loss:
      mean_i mean_{kNN} || z_i - h_j ||^2

    knn_k=1 时等价于 nearest neighbor (paper-like)
    """
    z = z.float()
    H_ref = H_ref.float()

    # dist^2 = ||z||^2 + ||h||^2 - 2 z h^T
    z2 = (z * z).sum(dim=1, keepdim=True)        # (B,1)
    h2 = (H_ref * H_ref).sum(dim=1).unsqueeze(0) # (1,M)
    dist2 = z2 + h2 - 2.0 * (z @ H_ref.t())      # (B,M)
    dist2 = torch.clamp(dist2, min=0.0)

    k = int(knn_k)
    if k < 1:
        k = 1
    k = min(k, dist2.shape[1])

    vals, _ = torch.topk(dist2, k=k, dim=1, largest=False, sorted=False)  # (B,k)
    return vals.mean()