@echo off
REM ===================================================================
REM train_v2_3090.bat  —  V2 data format training on RTX 3090 (cuda:0)
REM Skills: mul_2d, div_1d, sub_2d_borrow
REM New format: example-first CoT, answer-at-end (#### final line only)
REM ===================================================================

cd /d F:\projects\autoresearch
if not exist math_lab\results\logs mkdir math_lab\results\logs
set CUDA_VISIBLE_DEVICES=0
set MLFLOW_EXPERIMENT=v2_3090

REM --- mul_2d: was 93.2%% — harder skill, more epochs ---
echo ============================================================
echo  MUL_2D v2  (3090)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\mul_2d_rich_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\mul_2d_rich_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_mul2d ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 800 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.05 --label-smoothing 0.05 --warmup 100 ^
  --val-every 50 --save-every 100 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_mul2d.stdout.log ^
  2>> math_lab\results\logs\v2_mul2d.stderr.log

REM --- div_1d: was 96.9%% — small dataset, needs epochs ---
echo ============================================================
echo  DIV_1D v2  (3090)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\div_1d_richcot_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\div_1d_richcot_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_div1d ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 800 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.0 --label-smoothing 0.05 --warmup 100 ^
  --val-every 50 --save-every 100 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_div1d.stdout.log ^
  2>> math_lab\results\logs\v2_div1d.stderr.log

REM --- sub_2d_borrow: was 100%% — verify format change maintains it ---
echo ============================================================
echo  SUB_2D_BORROW v2  (3090)
echo ============================================================
py -3.12 math_lab\finetune.py ^
  --data     math_lab\results\skills\sub_2d_borrow_rich_train_seed42_20260502_155518.jsonl ^
  --val-data math_lab\results\skills\sub_2d_borrow_rich_val_seed42_20260502_155518.jsonl ^
  --ckpt-dir math_lab\results\checkpoints\20260502_v2_sub2d_borrow ^
  --n-layer 6 --n-head 8 --n-embd 256 --seq-len 256 ^
  --epochs 600 --batch 32 --lr 6e-4 --lr-min 1e-5 ^
  --weight-decay 1.0 --dropout 0.05 --label-smoothing 0.1 --warmup 100 ^
  --val-every 50 --save-every 100 --num-workers 0 --em-every 0 ^
  --cooldown 25 --device cuda --seed 42 ^
  1>> math_lab\results\logs\v2_sub2d_borrow.stdout.log ^
  2>> math_lab\results\logs\v2_sub2d_borrow.stderr.log

echo V2_3090_DONE >> math_lab\results\logs\v2_mul2d.stdout.log
