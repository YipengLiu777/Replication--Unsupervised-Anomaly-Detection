@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo =========================================
echo [EXP009] eval 224 started
echo CWD: %CD%
echo =========================================

where python
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Please run: conda activate torch
  pause
  exit /b 1
)

set EXP_NAME=exp009_224_pure_unsup_ref
set RUN_DIR=runs\%EXP_NAME%
set CKPT=%RUN_DIR%\checkpoints\epoch_020.pt
set DATA=data\16000clean7dirty.pkl

echo.
echo RUN_DIR=%RUN_DIR%
echo CKPT=%CKPT%
echo DATA=%DATA%
echo.

if not exist "%CKPT%" (
  echo [ERROR] checkpoint not found:
  echo   %CKPT%
  echo Please check your exp name and epoch number.
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

echo.
echo ===== eval finished. EXITCODE=%ERRORLEVEL% =====
pause
endlocal