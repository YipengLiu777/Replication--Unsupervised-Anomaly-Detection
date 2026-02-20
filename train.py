import os, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import seed_everything, abspath, now_tag
from src.data import load_pkl, ImageDataset
from src.models import get_model
from src.ref_loss import build_href_from_clean, latent_ref_loss_knn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pkl", type=str, default="data/16000clean7dirty.pkl")
    ap.add_argument("--dirty_key", type=str, default="raindrop")
    ap.add_argument("--out_size", type=int, default=64, choices=[64, 224])
    ap.add_argument("--latent_dim", type=int, default=256)

    ap.add_argument("--n_clean", type=int, default=16000)
    ap.add_argument("--n_dirty", type=int, default=1600)
    ap.add_argument("--clean_only", action="store_true", help="train only clean samples (no dirty in training data)")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--exp_name", type=str, default="")

    # -------- EXP003 knobs (reference loss) --------
    ap.add_argument("--use_ref_loss", action="store_true", help="enable reference loss term")
    ap.add_argument("--lambda_ref", type=float, default=1.0, help="weight for reference loss")
    ap.add_argument("--m_ref", type=int, default=1024, help="size of reference set H_ref")
    ap.add_argument("--ref_batch", type=int, default=128, help="batch size when building H_ref")
    ap.add_argument("--href_refresh", type=str, default="every", choices=["once", "every"],
                    help="build H_ref once or rebuild every epoch (paper-like: every)")
    ap.add_argument("--knn_k", type=int, default=1, help="1=paper nearest neighbor; >1=kNN-mean (more stable)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (optional)")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = args.exp_name.strip() or f"exp_{now_tag()}_{args.out_size}_{'cleanonly' if args.clean_only else 'mixed'}"
    run_dir = os.path.join("runs", exp)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("device:", device)
    print("CWD:", os.getcwd())
    print("run_dir(abs):", abspath(run_dir))

    # data
    x_clean, x_dirty = load_pkl(args.data_pkl, args.dirty_key, args.n_clean, args.n_dirty)
    ds = ImageDataset(x_clean, x_dirty, out_size=args.out_size, include_dirty=(not args.clean_only))
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

    # Build H_ref (optional)
    H_ref = None
    if args.use_ref_loss and args.href_refresh == "once":
        # IMPORTANT: only from clean indices [0, n_clean)
        H_ref = build_href_from_clean(
            model=model,
            dataset=ds,
            n_clean=args.n_clean,
            M=args.m_ref,
            batch_ref=args.ref_batch,
            device=device,
        )

    # train
    for ep in range(1, args.epochs + 1):
        if args.use_ref_loss and args.href_refresh == "every":
            H_ref = build_href_from_clean(
                model=model,
                dataset=ds,
                n_clean=args.n_clean,
                M=args.m_ref,
                batch_ref=args.ref_batch,
                device=device,
            )

        model.train()
        run_rec = 0.0
        run_ref = 0.0
        run_total = 0.0
        steps = 0

        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)

            # reconstruction
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                xrec, z = model(xb)
                Lrec = F.mse_loss(xrec, xb)

            # reference loss in FP32
            if args.use_ref_loss:
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

        if args.use_ref_loss:
            print(f"[ep {ep:03d}] Lrec={avg_rec:.6f}  Lref={avg_ref:.3e}  L={avg_tot:.6f}")
        else:
            print(f"[ep {ep:03d}] mse={avg_rec:.6f}")

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt")
        torch.save(
            {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "args": vars(args)},
            ckpt_path,
        )

    print("done. last ckpt:", abspath(ckpt_path))


if __name__ == "__main__":
    main()