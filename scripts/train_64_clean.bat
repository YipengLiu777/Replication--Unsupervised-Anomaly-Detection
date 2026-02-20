@echo off
cd /d %~dp0\..
python -u train.py --clean_only --out_size 64 --epochs 20 --batch_size 64 --exp_name exp001_64_cleanonly