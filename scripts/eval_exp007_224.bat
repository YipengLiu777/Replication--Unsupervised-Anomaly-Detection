@echo off
cd /d %~dp0\..

set RUN_DIR=runs\exp007_224_mixed_ref_l50_k5_m2048
set CKPT=%RUN_DIR%\checkpoints\epoch_020.pt

python -u eval.py --out_size 224 --run_dir "%RUN_DIR%" --ckpt "%CKPT%" --strict