@echo off
setlocal

REM ===== EXP008 eval (use last ckpt) =====
python -u eval.py ^
  --data_pkl "data/16000clean7dirty.pkl" ^
  --dirty_key raindrop ^
  --out_size 224 ^
  --latent_dim 256 ^
  --n_clean 16000 ^
  --n_dirty 1600 ^
  --batch_size 256 ^
  --ckpt "runs\exp008_224_mixed_noref\checkpoints\epoch_020.pt" ^
  --run_dir "runs\exp008_224_mixed_noref" ^
  --strict

endlocal