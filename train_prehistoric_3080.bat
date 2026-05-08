@echo off
REM ===================================================================
REM train_prehistoric_3080.bat — PDP-11 SLM: 30,000–10,000 BCE era run
REM
REM Prerequisites:
REM   git pull  (to get sources_prehistoric/ and training/ from Mac)
REM
REM Architecture : Config A (7M params)
REM GPU           : RTX 3080 (cuda:1)
REM Data          : sources_prehistoric\fineweb-edu\ (31,482 rows)
REM Token budget  : 140M (Chinchilla for 7M model)
REM ===================================================================

cd /d F:\projects\autoresearch
if not exist training\checkpoints\prehistoric_config_a mkdir training\checkpoints\prehistoric_config_a
if not exist training\logs\prehistoric_config_a mkdir training\logs\prehistoric_config_a

set CUDA_VISIBLE_DEVICES=1
set PYTHONPATH=F:\projects\autoresearch\training

echo ============================================================
echo  PDP-11 SLM — PREHISTORIC ERA (30k-10k BCE)  [RTX 3080]
echo  Config A: 7M params ^| 140M tokens ^| bfloat16
echo  Data: sources_prehistoric\fineweb-edu\ (31,482 rows)
echo ============================================================

py -3.12 training\train.py ^
  --config A ^
  --data-dir sources_prehistoric\fineweb-edu ^
  --out-dir  training\checkpoints\prehistoric_config_a ^
  --resume ^
  1>> training\logs\prehistoric_config_a\train.stdout.log ^
  2>> training\logs\prehistoric_config_a\train.stderr.log

echo PREHISTORIC_A_DONE >> training\logs\prehistoric_config_a\train.stdout.log
echo.
echo Training finished. Logs at training\logs\prehistoric_config_a\train.stdout.log
