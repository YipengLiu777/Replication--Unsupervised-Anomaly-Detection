@echo off
setlocal EnableExtensions EnableDelayedExpansion

set EXP_NAME=exp009_224_pure_unsup_ref_l1
set RUN_DIR=runs\%EXP_NAME%
set CKPT=%RUN_DIR%\checkpoints\epoch_020.pt
set DATA=data\16000clean7dirty.pkl

if not exist "%CKPT%" (
  echo [ERROR] checkpoint not found:
  echo   %CKPT%
  pause
  exit /b 1
)

python -u eval.py ^
  --data_pkl "%DATA%" ^
  --dirty_key raindrop ^
  --out_size 224 ^
  --latent_dim 256 ^
  --n_clean 16000 ^
  --n_dirty 1600 ^
  --batch_size 256 ^
  --ckpt "%CKPT%" ^
  --run_dir "%RUN_DIR%" ^
  --strict

pause
endlocal