@echo off
cd /d %~dp0\..
python -u eval.py --out_size 64 --run_dir runs\exp002_64_mixed_noref --ckpt runs\exp002_64_mixed_noref\checkpoints\epoch_020.pt --strict