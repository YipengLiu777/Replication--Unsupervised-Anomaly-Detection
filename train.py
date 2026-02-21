import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import seed_everything, abspath, now_tag
from src.data import load_pkl, ImageDataset
from src.models import get_model
from src.ref_loss import build_href_from_clean, build_href_from_indices, latent_ref_loss_knn
from src.metrics import score_dataset


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_pkl", type=str, default="data/16000clean7dirty.pkl")
    ap.add_argument("--dirty_key", type=str, default="raindrop")
    ap.add_argument("--out_size", type=int, default=224, choices=[64, 224])
    ap.add_argument("--latent_dim", type=int, default=256)

    ap.add_argument("--n_clean", type=int, default=16000)
    ap.add_argument("--n_dirty", type=int, default=1600)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--exp_name", type=str, default="")

    # -------- reference loss knobs --------
    ap.add_argument("--use_ref_loss", action="store_true")
    ap.add_argument("--lambda_ref", type=float, default=50.0)
    ap.add_argument("--m_ref", type=int, default=2048)
    ap.add_argument("--ref_batch", type=int, default=128)
    ap.add_argument("--href_refresh", type=str, default="every", choices=["once", "every"])
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--amp", action="store_true")

    # -------- PURE UNSUPERVISED (EXP009) knobs --------
    ap.add_argument("--pure_unsup", action="store_true", help="enable pure unsupervised self-filter training")
    ap.add_argument("--warmup_epochs", type=int, default=2, help="train only Lrec for first N epochs")
    ap.add_argument("--filter_every", type=int, default=1, help="recompute pseudo-clean every k epochs")
    ap.add_argument("--keep_frac", type=float, default=0.95, help="fraction kept as pseudo-clean")
    ap.add_argument("--score_metric", type=str, default="pcc", choices=["pcc", "mse"])
    ap.add_argument("--score_bs", type=int, default=256, help="batch size for scoring pass")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = args.exp_name.strip() or f"exp_{now_tag()}_{args.out_size}_unsup" if args.pure_unsup else f"exp_{now_tag()}_{args.out_size}_sup"
    run_dir = os.path.join("runs", exp)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("device:", device)
    print("CWD:", os.getcwd())
    print("run_dir(abs):", abspath(run_dir))

    # data (we load and construct mixed dataset; training will decide how to use it)
    x_clean, x_dirty = load_pkl(args.data_pkl, args.dirty_key, args.n_clean, args.n_dirty)
    ds = ImageDataset(x_clean, x_dirty, out_size=args.out_size, include_dirty=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # model + optimizer
    model = get_model(args.out_size, args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    N_total = len(ds)
    all_indices = np.arange(N_total)

    # init reference set
    H_ref = None

    for ep in range(1, args.epochs + 1):

        # -------- EXP009: pure unsupervised self-filter to build H_ref --------
        if args.pure_unsup and args.use_ref_loss:
            if ep <= args.warmup_epochs:
                use_ref_this_epoch = False
            else:
                use_ref_this_epoch = True

            # recompute pseudo-clean and H_ref every filter_every epochs after warmup
            if use_ref_this_epoch and ((ep - args.warmup_epochs - 1) % args.filter_every == 0 or H_ref is None):
                scores = score_dataset(
                    model=model,
                    dataset=ds,
                    batch_size=args.score_bs,
                    device=device,
                    metric=args.score_metric,
                )
                keep = int(max(1, round(args.keep_frac * N_total)))

                if args.score_metric == "mse":
                    # mse low = normal
                    keep_idx = np.argsort(scores)[:keep]
                else:
                    # pcc high = normal
                    keep_idx = np.argsort(scores)[-keep:]

                H_ref = build_href_from_indices(
                    model=model,
                    dataset=ds,
                    indices=keep_idx,
                    M=args.m_ref,
                    batch_ref=args.ref_batch,
                    device=device,
                )
                print(f"[ep {ep:03d}] pseudo-clean kept={keep}/{N_total} ({keep/N_total:.3f}), metric={args.score_metric}")

        else:
            # -------- your old (semi-supervised) behavior: reference from known clean --------
            use_ref_this_epoch = args.use_ref_loss
            if use_ref_this_epoch:
                if args.href_refresh == "every" or H_ref is None:
                    H_ref = build_href_from_clean(
                        model=model,
                        dataset=ds,
                        n_clean=args.n_clean,
                        M=args.m_ref,
                        batch_ref=args.ref_batch,
                        device=device,
                    )

        # -------- train one epoch --------
        model.train()
        run_rec = 0.0
        run_ref = 0.0
        run_total = 0.0
        steps = 0

        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                xrec, z = model(xb)
                Lrec = F.mse_loss(xrec, xb)

            if use_ref_this_epoch:
                with torch.cuda.amp.autocast(enabled=False):
                    Lref = latent_ref_loss_knn(z, H_ref, knn_k=args.knn_k)
                loss = Lrec + args.lambda_ref * Lref
            else:
                Lref = torch.zeros((), dtype=torch.float32, device=device)
                loss = Lrec

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            run_rec += float(Lrec.item())
            run_ref += float(Lref.item())
            run_total += float(loss.item())
            steps += 1

        avg_rec = run_rec / max(1, steps)
        avg_ref = run_ref / max(1, steps)
        avg_tot = run_total / max(1, steps)

        if use_ref_this_epoch:
            print(f"[ep {ep:03d}] Lrec={avg_rec:.6f}  Lref={avg_ref:.3e}  L={avg_tot:.6f}")
        else:
            print(f"[ep {ep:03d}] mse={avg_rec:.6f} (warmup/no-ref)")

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt")
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args)}, ckpt_path)

    print("done. last ckpt:", abspath(ckpt_path))


if __name__ == "__main__":
    main()