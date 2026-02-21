@echo off
setlocal

python -u train.py ^
  --data_pkl "data/16000clean7dirty.pkl" ^
  --dirty_key raindrop ^
  --out_size 224 ^
  --latent_dim 256 ^
  --n_clean 16000 ^
  --n_dirty 1600 ^
  --epochs 20 ^
  --batch_size 32 ^
  --lr 5e-4 ^
  --seed 42 ^
  --exp_name exp009_224_pure_unsup_ref ^
  --use_ref_loss ^
  --lambda_ref 50 ^
  --m_ref 2048 ^
  --ref_batch 128 ^
  --knn_k 5 ^
  --href_refresh every ^
  --pure_unsup ^
  --warmup_epochs 2 ^
  --filter_every 1 ^
  --keep_frac 0.95 ^
  --score_metric pcc ^
  --score_bs 256

endlocal