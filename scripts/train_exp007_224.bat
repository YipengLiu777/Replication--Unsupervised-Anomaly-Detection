@echo off
cd /d %~dp0\..

REM ===== EXP007: 224x224, mixed + ref loss (paper-aligned size) =====
REM If OOM, reduce batch_size to 16 or 8, and/or reduce m_ref to 1024.

python -u train.py ^
  --out_size 224 ^
  --epochs 20 ^
  --batch_size 16 ^
  --lr 5e-4 ^
  --exp_name exp007_224_mixed_ref_l50_k5_m2048 ^
  --use_ref_loss ^
  --lambda_ref 50 ^
  --knn_k 5 ^
  --m_ref 2048 ^
  --ref_batch 64 ^
  --href_refresh every