@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo =========================================
echo [EXP009] PURE UNSUP train 224 lambda=1
echo CWD: %CD%
echo =========================================

where python
if errorlevel 1 (
  echo [ERROR] python not found in PATH. Please run: conda activate torch
  pause
  exit /b 1
)
python --version

set EXP_NAME=exp009_224_pure_unsup_ref_l1
set DATA=data\16000clean7dirty.pkl

echo.
echo EXP_NAME=%EXP_NAME%
echo DATA=%DATA%
echo.

python -u train.py ^
  --data_pkl "%DATA%" ^
  --dirty_key raindrop ^
  --out_size 224 ^
  --latent_dim 256 ^
  --n_clean 16000 ^
  --n_dirty 1600 ^
  --epochs 20 ^
  --batch_size 32 ^
  --lr 5e-4 ^
  --seed 42 ^
  --exp_name "%EXP_NAME%" ^
  --use_ref_loss ^
  --lambda_ref 1 ^
  --m_ref 2048 ^
  --ref_batch 128 ^
  --href_refresh every ^
  --knn_k 5 ^
  --pure_unsup ^
  --warmup_epochs 2 ^
  --filter_every 1 ^
  --keep_frac 0.95 ^
  --score_metric pcc ^
  --score_bs 256

echo.
echo ===== train finished. EXITCODE=%ERRORLEVEL% =====
pause
endlocal