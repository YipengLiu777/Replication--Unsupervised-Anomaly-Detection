import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def _safe_load_npz(run_dir: str):
    npz_path = os.path.join(run_dir, "results", "eval_metrics.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing: {npz_path}\nDid you run eval.py for this run_dir?")
    d = np.load(npz_path, allow_pickle=True)
    return d, npz_path


def _try_load_report(run_dir: str):
    path = os.path.join(run_dir, "results", "eval_report.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _get_threshold(d, mode: str):
    """
    mode:
      - 'pct'   : use delta_pcc_pct (recommended)
      - 'paper' : use delta_pcc_paper
      - 'none'  : no PCC threshold line
    """
    if mode == "pct" and "delta_pcc_pct" in d.files:
        return float(d["delta_pcc_pct"]), "PCC delta (clean-percentile)"
    if mode == "paper" and "delta_pcc_paper" in d.files:
        return float(d["delta_pcc_paper"]), "PCC delta (paper-offset)"
    return None, None


def _hist_overlay(ax, clean, dirty, title, xlabel, bins=60, vline=None, vline_label=None):
    ax.hist(clean, bins=bins, alpha=0.55, density=True, label="clean")
    ax.hist(dirty, bins=bins, alpha=0.55, density=True, label="dirty")
    if vline is not None:
        ax.axvline(vline, linestyle="--", linewidth=1.2, label=vline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)


def _curve_by_index(ax, y, n_clean, title, ylabel, hline=None, hline_label=None, invert_keep=False):
    """
    invert_keep:
      False: keep if y >= hline (PCC-like)
      True : keep if y <= hline (MSE-like)
    """
    N = len(y)
    x = np.arange(N)
    ax.plot(x, y, linewidth=1.0, alpha=0.9)

    # clean/dirty boundary + dirty shading
    ax.axvline(n_clean, linestyle="--", linewidth=1.2)
    ax.axvspan(n_clean, N, alpha=0.08)

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.2, label=hline_label)

        # optional: show kept ratios on plot (very informative)
        if not np.isnan(hline):
            if invert_keep:
                keep_clean = (y[:n_clean] <= hline).mean()
                keep_dirty = (y[n_clean:] <= hline).mean()
            else:
                keep_clean = (y[:n_clean] >= hline).mean()
                keep_dirty = (y[n_clean:] >= hline).mean()
            ax.text(
                0.01, 0.02,
                f"kept clean={keep_clean:.3f}\nkept dirty={keep_dirty:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", alpha=0.12)
            )

    ax.set_title(title)
    ax.set_xlabel("Image Index (clean first, then raindrop)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if hline is not None:
        ax.legend(loc="best", framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help=r"e.g. runs\exp007_224_mixed_ref_l50_k5_m2048")
    ap.add_argument("--out", type=str, default="", help="output png path. default: <run_dir>/results/raindrop_viz.png")
    ap.add_argument("--pcc_mode", type=str, default="pct", choices=["pct", "paper", "none"],
                    help="Which PCC threshold to draw: pct (recommended), paper, none")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--downsample", type=int, default=0, help="0=off. e.g. 4000 to downsample index curves for speed")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    d, npz_path = _safe_load_npz(args.run_dir)
    report = _try_load_report(args.run_dir)

    # required arrays
    pcc = d["pcc"].astype(np.float64)
    pcc_med = d["pcc_med"].astype(np.float64)
    mse = d["mse"].astype(np.float64)
    n_clean = int(d["N_CLEAN"])
    n_dirty = int(d["N_DIRTY"])
    N = len(pcc_med)

    assert n_clean + n_dirty == N, f"N mismatch: n_clean({n_clean})+n_dirty({n_dirty}) != len(scores)({N})"

    # thresholds
    delta_pcc, delta_pcc_name = _get_threshold(d, args.pcc_mode)
    delta_mse = float(d["delta_mse"]) if "delta_mse" in d.files else None

    # downsample index curves (only curves; hist uses full)
    if args.downsample and args.downsample > 0 and N > args.downsample:
        step = max(1, N // args.downsample)
        idx = np.arange(0, N, step)
        pcc_med_curve = pcc_med[idx]
        mse_curve = mse[idx]
    else:
        pcc_med_curve = pcc_med
        mse_curve = mse

    # split clean/dirty
    clean_pccm = pcc_med[:n_clean]
    dirty_pccm = pcc_med[n_clean:]
    clean_mse = mse[:n_clean]
    dirty_mse = mse[n_clean:]

    # figure
    fig = plt.figure(figsize=(12.5, 8.5))
    fig.suptitle(f"Raindrop Visualization — {os.path.basename(args.run_dir)}", fontsize=14)

    ax1 = plt.subplot(2, 2, 1)
    _curve_by_index(
        ax1,
        pcc_med_curve,
        n_clean=n_clean if len(pcc_med_curve) == N else int(n_clean / (N / len(pcc_med_curve))),
        title="Median-filtered PCC by Index (Raindrop)",
        ylabel="PCC_med",
        hline=delta_pcc,
        hline_label=delta_pcc_name,
        invert_keep=False
    )

    ax2 = plt.subplot(2, 2, 2)
    _hist_overlay(
        ax2,
        clean_pccm,
        dirty_pccm,
        title="PCC_med Distribution (clean vs raindrop)",
        xlabel="PCC_med",
        bins=args.bins,
        vline=delta_pcc,
        vline_label=delta_pcc_name
    )

    ax3 = plt.subplot(2, 2, 3)
    _curve_by_index(
        ax3,
        mse_curve,
        n_clean=n_clean if len(mse_curve) == N else int(n_clean / (N / len(mse_curve))),
        title="MSE by Index (Raindrop)",
        ylabel="MSE",
        hline=delta_mse,
        hline_label="MSE delta (clean-percentile)",
        invert_keep=True
    )

    ax4 = plt.subplot(2, 2, 4)
    _hist_overlay(
        ax4,
        clean_mse,
        dirty_mse,
        title="MSE Distribution (clean vs raindrop)",
        xlabel="MSE",
        bins=args.bins,
        vline=delta_mse,
        vline_label="MSE delta (clean-percentile)"
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path = args.out.strip()
    if not out_path:
        out_path = os.path.join(args.run_dir, "results", "raindrop_viz.png")

    plt.savefig(out_path, dpi=220)
    print("Loaded:", npz_path)
    if report:
        print("Report:", os.path.join(args.run_dir, "results", "eval_report.json"))
    print("Saved figure:", out_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()