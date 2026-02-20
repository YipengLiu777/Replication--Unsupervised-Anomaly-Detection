# eval.py
# Evaluate clean vs dirty separation for a trained CAE.
# Threshold is controlled ONLY by constants below (no terminal tuning).

import os
import json
import argparse
import warnings
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_pkl, ImageDataset
from src.models import get_model
from src.metrics import compute_pcc_mse, median_filter_1d

# =========================
# THRESHOLD CONFIG (EDIT HERE)
# =========================

# 1) Paper-like PCC threshold (for reporting / alignment with paper)
#    delta_pcc_paper = median(clean_pcc_med) - PCC_DELTA_OFFSET
PCC_DELTA_OFFSET = 0.05

# 2) Percentile-based PCC threshold (recommended for your 64x64 runs)
#    We set delta so that about PCC_KEEP_CLEAN fraction of CLEAN are kept:
#    keep_clean ≈ PCC_KEEP_CLEAN, i.e., delta = percentile(clean, (1-keep)*100)
PCC_USE_PERCENTILE = True
PCC_KEEP_CLEAN = 0.95     # try 0.95, 0.99 (more strict), 0.90 (looser)

# MSE threshold: delta_mse = percentile(clean_mse, MSE_PERCENTILE)
MSE_PERCENTILE = 95.0

# Median filter window for PCC smoothing (paper uses ~100; we use 101 for odd)
MED_K = 101

# Numerical safety
EPS = 1e-12


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _pcc_delta_percentile(clean_pcc_med: np.ndarray, keep_clean: float) -> float:
    keep_clean = float(keep_clean)
    keep_clean = min(max(keep_clean, 0.5), 0.999)  # clamp to sane range
    q = (1.0 - keep_clean) * 100.0
    return float(np.percentile(clean_pcc_med, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pkl", type=str, default="data/16000clean7dirty.pkl")
    ap.add_argument("--dirty_key", type=str, default="raindrop")
    ap.add_argument("--out_size", type=int, default=64, choices=[64, 224])
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--n_clean", type=int, default=16000)
    ap.add_argument("--n_dirty", type=int, default=1600)
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -------- data (eval uses clean + dirty) --------
    x_clean, x_dirty = load_pkl(args.data_pkl, args.dirty_key, args.n_clean, args.n_dirty)
    ds = ImageDataset(x_clean, x_dirty, out_size=args.out_size, include_dirty=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # -------- load model --------
    model_cpu = get_model(args.out_size, args.latent_dim).cpu()

    warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing key 'model'. keys={list(ckpt.keys())}")

    model_cpu.load_state_dict(ckpt["model"], strict=args.strict)
    print(f"load_state_dict OK (strict={args.strict})")

    model = model_cpu.to(device)

    # -------- compute metrics --------
    pcc, mse = compute_pcc_mse(model, dl, device=device)
    pcc_med = median_filter_1d(pcc, k=MED_K)

    nC = int(args.n_clean)
    nD = int(args.n_dirty)

    clean_p = pcc_med[:nC]
    dirty_p = pcc_med[nC : nC + nD]
    clean_m = mse[:nC]
    dirty_m = mse[nC : nC + nD]

    # =========================
    # PCC thresholds
    # =========================
    # (A) paper-like offset
    delta_pcc_paper = float(np.median(clean_p) - PCC_DELTA_OFFSET)
    keep_paper = (pcc_med >= delta_pcc_paper).astype(np.uint8)
    kept_clean_paper = float(keep_paper[:nC].mean())
    kept_dirty_paper = float(keep_paper[nC : nC + nD].mean())

    print("---- PCC paper_offset ----")
    print("PCC_DELTA_OFFSET:", PCC_DELTA_OFFSET)
    print("delta_pcc_paper:", delta_pcc_paper)
    print("kept clean ratio (pcc paper):", kept_clean_paper)
    print("kept dirty ratio (pcc paper):", kept_dirty_paper)
    print("median PCC_med clean:", float(np.median(clean_p)))
    print("median PCC_med dirty:", float(np.median(dirty_p)))

    # (B) percentile threshold (recommended)
    delta_pcc_pct = None
    keep_pct = None
    kept_clean_pct = None
    kept_dirty_pct = None

    if PCC_USE_PERCENTILE:
        delta_pcc_pct = _pcc_delta_percentile(clean_p, PCC_KEEP_CLEAN)
        keep_pct = (pcc_med >= delta_pcc_pct).astype(np.uint8)
        kept_clean_pct = float(keep_pct[:nC].mean())
        kept_dirty_pct = float(keep_pct[nC : nC + nD].mean())

        print("---- PCC clean_percentile ----")
        print("PCC_KEEP_CLEAN:", PCC_KEEP_CLEAN)
        print("delta_pcc_pct:", float(delta_pcc_pct))
        print("kept clean ratio (pcc pct):", float(kept_clean_pct))
        print("kept dirty ratio (pcc pct):", float(kept_dirty_pct))

    # =========================
    # MSE threshold (clean p95)
    # =========================
    delta_mse = float(np.percentile(clean_m, MSE_PERCENTILE))
    keep_mse = (mse <= delta_mse).astype(np.uint8)
    kept_clean_mse = float(keep_mse[:nC].mean())
    kept_dirty_mse = float(keep_mse[nC : nC + nD].mean())

    print("---- MSE percentile ----")
    print("MSE_PERCENTILE:", MSE_PERCENTILE)
    print("delta_mse:", delta_mse)
    print("kept clean ratio (mse):", kept_clean_mse)
    print("kept dirty ratio (mse):", kept_dirty_mse)

    # -------- save results --------
    res_dir = os.path.join(args.run_dir, "results")
    _ensure_dir(res_dir)

    npz_path = os.path.join(res_dir, "eval_metrics.npz")
    np.savez(
        npz_path,
        pcc=pcc.astype(np.float32),
        pcc_med=pcc_med.astype(np.float32),
        mse=mse.astype(np.float32),

        # thresholds
        delta_pcc_paper=np.float32(delta_pcc_paper),
        delta_mse=np.float32(delta_mse),

        # masks
        keep_mask_pcc_paper=keep_paper,
        keep_mask_mse=keep_mse,

        # percentile PCC (optional)
        PCC_USE_PERCENTILE=np.int32(1 if PCC_USE_PERCENTILE else 0),
        PCC_KEEP_CLEAN=np.float32(PCC_KEEP_CLEAN),
        delta_pcc_pct=np.float32(delta_pcc_pct) if delta_pcc_pct is not None else np.float32(-1.0),
        keep_mask_pcc_pct=keep_pct if keep_pct is not None else np.zeros_like(keep_paper),

        N_CLEAN=np.int32(nC),
        N_DIRTY=np.int32(nD),
        out_size=np.int32(args.out_size),
        latent_dim=np.int32(args.latent_dim),
        dirty_key=args.dirty_key,
        MED_K=np.int32(MED_K),
        PCC_DELTA_OFFSET=np.float32(PCC_DELTA_OFFSET),
        MSE_PERCENTILE=np.float32(MSE_PERCENTILE),
    )
    print("saved:", npz_path)

    report = dict(
        ckpt=args.ckpt,
        run_dir=args.run_dir,
        dirty_key=args.dirty_key,
        out_size=args.out_size,
        latent_dim=args.latent_dim,
        n_clean=nC,
        n_dirty=nD,

        # PCC paper
        PCC_DELTA_OFFSET=float(PCC_DELTA_OFFSET),
        MED_K=int(MED_K),
        delta_pcc_paper=float(delta_pcc_paper),
        median_pcc_med_clean=float(np.median(clean_p)),
        median_pcc_med_dirty=float(np.median(dirty_p)),
        kept_clean_ratio_pcc_paper=float(kept_clean_paper),
        kept_dirty_ratio_pcc_paper=float(kept_dirty_paper),

        # PCC percentile
        PCC_USE_PERCENTILE=bool(PCC_USE_PERCENTILE),
        PCC_KEEP_CLEAN=float(PCC_KEEP_CLEAN),
        delta_pcc_pct=float(delta_pcc_pct) if delta_pcc_pct is not None else None,
        kept_clean_ratio_pcc_pct=float(kept_clean_pct) if kept_clean_pct is not None else None,
        kept_dirty_ratio_pcc_pct=float(kept_dirty_pct) if kept_dirty_pct is not None else None,

        # MSE
        MSE_PERCENTILE=float(MSE_PERCENTILE),
        delta_mse=float(delta_mse),
        kept_clean_ratio_mse=float(kept_clean_mse),
        kept_dirty_ratio_mse=float(kept_dirty_mse),
    )
    json_path = os.path.join(res_dir, "eval_report.json")
    _save_json(json_path, report)
    print("saved:", json_path)


if __name__ == "__main__":
    main()