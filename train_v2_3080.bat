@echo off
REM ===================================================================
REM train_v2_3080.bat  —  V2 data format training on RTX 3080 (cuda:1)
REM Skills: add_1d, add_2d, mul_1d, mod_1d, sub_2d_no_borrow
REM New format: example-first CoT, answer-at-end (#### final line only)
REM ===================================================================

cd /d F:\projects\autoresearch
if not exist math_lab\results\logs mkdir math_lab\results\logs
set CUDA_VISIBLE_DEVICES=1
set MLFLOW_EXPERIMENT=v2_3080

REM --- add_1d: full table memorization ---
echo ============================================================
echo  ADD_1D v2  (3080)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\add_1d_full_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\add_1d_full_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_add1d ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 300 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.0 --label-smoothing 0.05 --warmup 50 ^
  --val-every 25 --save-every 50 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_add1d.stdout.log ^
  2>> math_lab\results\logs\v2_add1d.stderr.log

REM --- add_2d: was 100%% — large dataset, standard recipe ---
echo ============================================================
echo  ADD_2D v2  (3080)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\add_2d_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\add_2d_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_add2d ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 300 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.05 --label-smoothing 0.1 --warmup 100 ^
  --val-every 25 --save-every 50 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_add2d.stdout.log ^
  2>> math_lab\results\logs\v2_add2d.stderr.log

REM --- mul_1d: full times-table ---
echo ============================================================
echo  MUL_1D v2  (3080)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\mul_1d_rich_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\mul_1d_rich_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_mul1d ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 300 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.0 --label-smoothing 0.05 --warmup 50 ^
  --val-every 25 --save-every 50 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_mul1d.stdout.log ^
  2>> math_lab\results\logs\v2_mul1d.stderr.log

REM --- mod_1d: full modulo table ---
echo ============================================================
echo  MOD_1D v2  (3080)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\mod_1d_full_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\mod_1d_full_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_mod1d ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 800 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.0 --label-smoothing 0.05 --warmup 100 ^
  --val-every 50 --save-every 100 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_mod1d.stdout.log ^
  2>> math_lab\results\logs\v2_mod1d.stderr.log

REM --- sub_2d_no_borrow: was 100%% — verify format holds ---
echo ============================================================
echo  SUB_2D_NO_BORROW v2  (3080)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\sub_2d_no_borrow_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\sub_2d_no_borrow_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_sub2d_no_borrow ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 300 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.05 --label-smoothing 0.1 --warmup 100 ^
  --val-every 25 --save-every 50 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_sub2d_no_borrow.stdout.log ^
  2>> math_lab\results\logs\v2_sub2d_no_borrow.stderr.log

echo V2_3080_DONE >> math_lab\results\logs\v2_add1d.stdout.log
