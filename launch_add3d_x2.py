import subprocess, os
base = r'F:\projects\autoresearch'
log  = base + r'\math_lab\results\logs'
ckpt = base + r'\math_lab\results\checkpoints'
skl  = base + r'\math_lab\results\skills'
ft   = base + r'\math_lab\finetune.py'
py   = r'C:\Python312\python.exe'
os.makedirs(ckpt + r'\20260503_v3_add3d_x2', exist_ok=True)

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = '0'
env['MLFLOW_EXPERIMENT'] = 'v3b_3090'
cmd = [py, ft,
    '--data',         skl + r'\add_3d_train_seed42_20260503_121952.jsonl',
    '--val-data',     skl + r'\add_3d_val_seed42_20260503_121952.jsonl',
    '--ckpt-dir',     ckpt + r'\20260503_v3_add3d_x2',
    '--n-layer',  '6', '--n-head', '8', '--n-embd', '256', '--seq-len', '256',
    '--epochs',  '1000', '--batch', '32', '--lr', '6e-4', '--lr-min', '1e-5',
    '--weight-decay', '1.0', '--dropout', '0.05',
    '--label-smoothing', '0.05', '--warmup', '100',
    '--val-every', '100', '--save-every', '200',
    '--num-workers', '0', '--em-every', '0',
    '--cooldown', '30', '--device', 'cuda', '--seed', '42',
]
print('Starting v3_add3d_x2 on GPU 0 (1000 epochs)...', flush=True)
out_f = open(log + r'\v3_add3d_x2.stdout.log', 'w')
err_f = open(log + r'\v3_add3d_x2.stderr.log', 'w')
subprocess.run(cmd, env=env, stdout=out_f, stderr=err_f)
out_f.close(); err_f.close()
print('Done: v3_add3d_x2', flush=True)
