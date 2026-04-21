"""
colab_train.py — MathGPT Training Runner for Google Colab

Paste this entire file into a Colab cell, or upload to Drive and run:
    !python /content/drive/MyDrive/autoresearch/math_lab/colab_train.py

Colab T4 (free tier):  16GB VRAM -> n_embd=512, n_layer=8, batch=128
Colab A100 (Colab Pro): 40GB VRAM -> n_embd=768, n_layer=12, batch=256

Prerequisites: run cells 1-3 below, then run the training cell.
"""

# ===========================================================================
# CELL 1 — Mount Drive and clone/pull repo
# ===========================================================================
CELL_1 = """
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess

REPO_URL  = "https://github.com/sj-laymaninnovations/autoresearch.git"
REPO_PATH = "/content/autoresearch"

if not os.path.exists(REPO_PATH):
    subprocess.run(["git", "clone", REPO_URL, REPO_PATH], check=True)
else:
    subprocess.run(["git", "-C", REPO_PATH, "pull"], check=True)

os.chdir(REPO_PATH)
print("Working directory:", os.getcwd())
"""

# ===========================================================================
# CELL 2 — Install deps (torch already on Colab; only need these extras)
# ===========================================================================
CELL_2 = """
# torch + CUDA are pre-installed on Colab — nothing else needed for finetune.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
print(f"GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB" if torch.cuda.is_available() else "")
"""

# ===========================================================================
# CELL 3 — Generate training data (5,000+ pairs for grokking threshold)
# ===========================================================================
CELL_3 = """
import subprocess, os
os.chdir("/content/autoresearch")

# Generate enough data to hit the grokking threshold (Power et al. 2022: need ~5k pairs)
runs = [
    ["python", "math_lab/datagen/harder_qa.py", "--domain", "arithmetic",    "--n", "2000"],
    ["python", "math_lab/datagen/harder_qa.py", "--domain", "arithmetic",    "--n", "1000", "--difficulty", "structural"],
    ["python", "math_lab/datagen/harder_qa.py", "--domain", "arithmetic",    "--n", "500",  "--difficulty", "adversarial"],
    ["python", "math_lab/datagen/harder_qa.py", "--domain", "number_theory", "--n", "800"],
    ["python", "math_lab/datagen/harder_qa.py", "--domain", "algebra",       "--n", "500"],
    ["python", "math_lab/datagen/harder_qa.py", "--domain", "combinatorics", "--n", "300"],
]
for cmd in runs:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-200:] if result.stdout else result.stderr[-200:])

# Merge and deduplicate
subprocess.run(["python", "math_lab/merge_results.py"], check=True)
"""

# ===========================================================================
# CELL 4 — Check GPU and pick model size accordingly
# ===========================================================================
CELL_4 = """
import torch

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
print(f"VRAM: {vram_gb:.1f} GB")

# T4 (15GB) -> medium config
# A100 (40GB) -> large config
# CPU fallback -> tiny config
if vram_gb >= 35:
    config = dict(n_layer=12, n_head=8, n_embd=768, batch=256, epochs=300)
    label  = "A100 Large (~85M params)"
elif vram_gb >= 12:
    config = dict(n_layer=8,  n_head=8, n_embd=512, batch=128, epochs=300)
    label  = "T4 Medium (~22M params)"
elif vram_gb >= 4:
    config = dict(n_layer=6,  n_head=4, n_embd=256, batch=64,  epochs=200)
    label  = "RTX 3050Ti Small (~5M params)"
else:
    config = dict(n_layer=4,  n_head=4, n_embd=128, batch=16,  epochs=100)
    label  = "CPU Tiny (~0.8M params)"

print(f"Selected config: {label}")
print(f"Config: {config}")
"""

# ===========================================================================
# CELL 5 — Train (uses advisor-recommended hyperparams from literature)
# ===========================================================================
CELL_5 = """
import subprocess, os
os.chdir("/content/autoresearch")

# Config is set by Cell 4 above.
# Key hyperparams from training_advisor.py (Power 2022, Nye 2021, Lee 2023):
#   --weight-decay 1.0    (Grokking paper: essential for arithmetic generalization)
#   --dropout 0.15        (Regularization for small dataset)
#   --lr 1e-3             (Arithmetic-tuned LR from Lee et al.)
#   --label-smoothing 0.1 (Reduces memorization)

cmd = [
    "python", "math_lab/finetune.py",
    "--epochs",          str(config["epochs"]),
    "--batch",           str(config["batch"]),
    "--n-layer",         str(config["n_layer"]),
    "--n-head",          str(config["n_head"]),
    "--n-embd",          str(config["n_embd"]),
    "--seq-len",         "256",
    "--lr",              "1e-3",
    "--lr-min",          "1e-5",
    "--warmup",          "200",
    "--weight-decay",    "1.0",
    "--dropout",         "0.15",
    "--label-smoothing", "0.1",
    "--val-split",       "0.15",
    "--save-every",      "20",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, text=True)
"""

# ===========================================================================
# CELL 6 — Copy best checkpoint to Drive for safekeeping
# ===========================================================================
CELL_6 = """
import shutil, os
from pathlib import Path

CKPT_SRC  = Path("/content/autoresearch/math_lab/results/checkpoints/best.pt")
DRIVE_DST = Path("/content/drive/MyDrive/autoresearch_checkpoints/")
DRIVE_DST.mkdir(parents=True, exist_ok=True)

if CKPT_SRC.exists():
    dst = DRIVE_DST / "best.pt"
    shutil.copy2(CKPT_SRC, dst)
    print(f"Saved checkpoint -> {dst}")
    print(f"Size: {dst.stat().st_size / 1024**2:.1f} MB")
else:
    print("No checkpoint found at", CKPT_SRC)
"""

# ===========================================================================
# CELL 7 — Sample inference on Colab
# ===========================================================================
CELL_7 = """
import subprocess
result = subprocess.run(
    ["python", "math_lab/finetune.py",
     "--eval-only",
     "--ckpt", "math_lab/results/checkpoints/best.pt"],
    capture_output=True, text=True, cwd="/content/autoresearch"
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])
"""

# ===========================================================================
# Print instructions when this script is run directly
# ===========================================================================
if __name__ == "__main__":
    import textwrap

    instructions = """
    ===================================================================
     Google Colab Setup — MathGPT Training
    ===================================================================

     QUICK START (all cells as one paste):

     1. Open https://colab.research.google.com
     2. Runtime -> Change runtime type -> T4 GPU
     3. New notebook, paste each CELL below into separate code cells
     4. Run cells 1 through 7 in order

     EXPECTED TIMES (T4 GPU, 300 epochs, 5k pairs):
       Data generation : ~30 seconds
       Training        : ~8-12 minutes
       Total session   : ~15 minutes
       Checkpoint size : ~20-80 MB depending on config

     SAVING YOUR WORK:
       Cell 6 copies best.pt to your Google Drive automatically.
       To resume on your local RTX 3050 Ti:
         python math_lab/finetune.py --ckpt /path/to/best.pt --epochs 50

     COLAB SESSION NOTES:
       - Free T4 sessions disconnect after ~12 hours idle
       - Save checkpoints to Drive (Cell 6) before closing
       - Pro ($10/mo) gives A100 access: ~3x faster than T4
    ===================================================================
    """
    print(textwrap.dedent(instructions))

    cells = {
        "CELL 1 — Mount Drive + Clone Repo": CELL_1,
        "CELL 2 — Verify GPU":               CELL_2,
        "CELL 3 — Generate Training Data":   CELL_3,
        "CELL 4 — Auto-select Model Config": CELL_4,
        "CELL 5 — Train":                    CELL_5,
        "CELL 6 — Save Checkpoint to Drive": CELL_6,
        "CELL 7 — Sample Inference":         CELL_7,
    }

    for title, code in cells.items():
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
        print(textwrap.dedent(code))
