@echo off
cd /d %~dp0\..

python -u train.py --out_size 64 --epochs 20 --batch_size 64 --exp_name exp004_64_mixed_ref_l10_k5 ^
  --use_ref_loss --lambda_ref 10 --m_ref 1024 --ref_batch 128 --href_refresh every --knn_k 5