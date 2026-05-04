import subprocess, os, sys
base = r'F:\projects\autoresearch'
log  = base + r'\math_lab\results\logs'
ckpt = base + r'\math_lab\results\checkpoints'
skl  = base + r'\math_lab\results\skills'
ft   = base + r'\math_lab\finetune.py'
py   = r'C:\Python312\python.exe'
os.makedirs(log, exist_ok=True)
os.makedirs(ckpt, exist_ok=True)

# GPU 0 (3090) — add_3d (gap fix) + alg_2step
skills = [
    ('v3_add3d',    0, 'add_3d_train_seed42_20260503_121952.jsonl',
                       'add_3d_val_seed42_20260503_121952.jsonl',
                       '20260503_v3_add3d', 600, 0.05, 0.05, 100, 50, 100),
    ('v3_alg2step', 0, 'alg_2step_train_seed42_20260503_121952.jsonl',
                       'alg_2step_val_seed42_20260503_121952.jsonl',
                       '20260503_v3_alg2step', 800, 0.05, 0.05, 100, 50, 100),
]

for name, cuda, tr, vl, ck, ep, do, ls, wu, ve, se in skills:
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(cuda)
    env['MLFLOW_EXPERIMENT'] = 'v3_3090'
    cmd = [py, ft,
        '--data',      os.path.join(skl, tr),
        '--val-data',  os.path.join(skl, vl),
        '--ckpt-dir',  os.path.join(ckpt, ck),
        '--n-layer',   '6', '--n-head', '8', '--n-embd', '256', '--seq-len', '256',
        '--epochs',    str(ep), '--batch', '32', '--lr', '6e-4', '--lr-min', '1e-5',
        '--weight-decay', '1.0', '--dropout', str(do),
        '--label-smoothing', str(ls), '--warmup', str(wu),
        '--val-every', str(ve), '--save-every', str(se),
        '--num-workers', '0', '--em-every', '0',
        '--cooldown', '25', '--device', 'cuda', '--seed', '42',
    ]
    print(f'Starting {name} on GPU {cuda} ({ep} epochs)...', flush=True)
    out_f = open(os.path.join(log, name + '.stdout.log'), 'w')
    err_f = open(os.path.join(log, name + '.stderr.log'), 'w')
    subprocess.run(cmd, env=env, stdout=out_f, stderr=err_f)
    out_f.close(); err_f.close()
    print(f'Done: {name}', flush=True)

print('v3_3090 ALL DONE', flush=True)
