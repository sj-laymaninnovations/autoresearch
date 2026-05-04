"""Track A (3080 GPU 1): div_1d_v4 (1500ep) then alg_1step_v2 (1000ep)."""
import subprocess, os
base = r'F:\projects\autoresearch'
log  = base + r'\math_lab\results\logs'
ckpt = base + r'\math_lab\results\checkpoints'
skl  = base + r'\math_lab\results\skills'
ft   = base + r'\math_lab\finetune.py'
py   = r'C:\Python312\python.exe'

SKILLS = [
    ('v4_div1d',      1, 'div_1d_v4_train_seed42_20260503_210946.jsonl',
                         'div_1d_v4_val_seed42_20260503_210946.jsonl',
                         '20260503_v4_div1d',      1500, 0.0, 0.05, 150, 100, 150),
    ('v4_alg1step',   1, 'alg_1step_v2_train_seed42_20260503_210946.jsonl',
                         'alg_1step_v2_val_seed42_20260503_210946.jsonl',
                         '20260503_v4_alg1step',   1000, 0.05, 0.05, 100, 100, 200),
]

for name, cuda, tr, vl, ck, ep, do, ls, wu, ve, se in SKILLS:
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = str(cuda); env['MLFLOW_EXPERIMENT'] = 'v4_A'
    cmd = [py, ft,
        '--data', os.path.join(skl,tr), '--val-data', os.path.join(skl,vl),
        '--ckpt-dir', os.path.join(ckpt,ck),
        '--n-layer','6','--n-head','8','--n-embd','256','--seq-len','256',
        '--epochs',str(ep),'--batch','32','--lr','6e-4','--lr-min','1e-5',
        '--weight-decay','1.0','--dropout',str(do),'--label-smoothing',str(ls),
        '--warmup',str(wu),'--val-every',str(ve),'--save-every',str(se),
        '--num-workers','0','--em-every','0','--cooldown','30','--device','cuda','--seed','42',
    ]
    print(f'[3080] {name} ({ep}ep)...', flush=True)
    of=open(os.path.join(log,name+'.stdout.log'),'w'); ef=open(os.path.join(log,name+'.stderr.log'),'w')
    subprocess.run(cmd, env=env, stdout=of, stderr=ef); of.close(); ef.close()
    print(f'Done: {name}', flush=True)
print('Track A DONE', flush=True)
