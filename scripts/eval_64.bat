@echo off
cd /d %~dp0\..
python -u eval.py --out_size 64 --run_dir runs\exp001_64_cleanonly --ckpt runs\exp001_64_cleanonly\checkpoints\epoch_020.pt --strict