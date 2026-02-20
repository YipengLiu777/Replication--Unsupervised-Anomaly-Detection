@echo off
REM eval_exp003_64.bat
REM Run evaluation for EXP003 (mixed + ref loss) at 64x64

cd /d %~dp0\..

set RUN_DIR=runs\exp003_64_mixed_ref
set CKPT=%RUN_DIR%\checkpoints\epoch_020.pt

python -u eval.py --out_size 64 --run_dir "%RUN_DIR%" --ckpt "%CKPT%" --strict