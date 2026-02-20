@echo off
cd /d %~dp0\..

python -u train.py --out_size 64 --epochs 20 --batch_size 64 --exp_name exp005_64_mixed_ref_l50_k5_m2048 ^
  --use_ref_loss --lambda_ref 50 --m_ref 2048 --ref_batch 128 --href_refresh every --knn_k 5