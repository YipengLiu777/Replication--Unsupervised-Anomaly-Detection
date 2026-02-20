@echo off
setlocal

REM ===== activate env (optional) =====
REM call D:\Anaconda3\Scripts\activate
REM conda activate torch

REM ===== EXP008: 224 mixed, NO ref loss =====
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
  --exp_name exp008_224_mixed_noref

endlocal