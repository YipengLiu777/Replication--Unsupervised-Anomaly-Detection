@echo off
cd /d %~dp0\..
python -u train.py --out_size 64 --epochs 20 --batch_size 64 --exp_name exp002_64_mixed_noref