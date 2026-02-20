@echo off
cd /d %~dp0\..
python -u train.py --out_size 64 --epochs 20 --batch_size 64 --exp_name exp003_64_mixed_ref ^
  --use_ref_loss --lambda_ref 1.0 --m_ref 1024 --ref_batch 128 --href_refresh every --knn_k 1