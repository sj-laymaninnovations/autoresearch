import subprocess, os, sys
base = r'F:\projects\autoresearch'
log  = base + r'\math_lab\results\logs'
ckpt = base + r'\math_lab\results\checkpoints'
skl  = base + r'\math_lab\results\skills'
ft   = base + r'\math_lab\finetune.py'
py   = r'C:\Python312\python.exe'
os.makedirs(log, exist_ok=True)
os.makedirs(ckpt, exist_ok=True)

skills = [
    ('v2_mul2d', 0, 'mul_2d_rich_train_seed42_20260502_155518.jsonl',
     'mul_2d_rich_val_seed42_20260502_155518.jsonl',
     '20260502_v2_mul2d', 800, 0.05, 0.05, 100, 50, 100),
    ('v2_div1d', 0, 'div_1d_richcot_train_seed42_20260502_155518.jsonl',
     'div_1d_richcot_val_seed42_20260502_155518.jsonl',
     '20260502_v2_div1d', 800, 0.0, 0.05, 100, 50, 100),
    ('v2_sub2d_borrow', 0, 'sub_2d_borrow_rich_train_seed42_20260502_155518.jsonl',
     'sub_2d_borrow_rich_val_seed42_20260502_155518.jsonl',
     '20260502_v2_sub2d_borrow', 600, 0.05, 0.1, 100, 50, 100),
]

for name, cuda, tr, vl, ck, ep, do, ls, wu, ve, se in skills:
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(cuda)
    env['MLFLOW_EXPERIMENT'] = 'v2_3090'
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

print('v2_3090 ALL DONE', flush=True)
