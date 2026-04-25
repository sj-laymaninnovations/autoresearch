"""
cot_teacher.py — Generate CoT step-by-step solutions via LM Studio teacher LLM

Distinct from cot_qa.py (which generates synthetic word problems with backward
verification). This module takes existing arithmetic problems and asks a
teacher model (e.g. OSS-20, Qwen 3.6) to produce step-by-step solutions,
then validates the teacher's final answer against the known-correct answer.

Usage:
    python -m math_lab.datagen.cot_teacher \\
        --input results/canonical_arith_train_t12master_*.jsonl \\
        --output results/canonical_arith_train_t12master_cot.jsonl \\
        --model openai/gpt-oss-20b --workers 4
"""

import argparse, json, re, time
from pathlib import Path
import concurrent.futures as cf
import requests

URL = "http://127.0.0.1:1234/v1/chat/completions"

PROMPT_TEMPLATE = """Compute step-by-step in compact form. Rules:
- Show 2-4 intermediate steps only (no LaTeX, no enumerations, no explanations)
- Use '=' between steps
- Each step on its own line
- Last line must be exactly '#### N' where N is the final numeric answer

Problem: {prob}"""


def call_teacher(prob, model, max_tokens=300, timeout=120):
    t0 = time.time()
    try:
        r = requests.post(URL, json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(prob=prob)}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }, timeout=timeout)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        return None, 0, time.time() - t0, str(e)
    content = d['choices'][0]['message']['content']
    tokens = d.get('usage', {}).get('completion_tokens', 0)
    return content, tokens, time.time() - t0, None


_ANS_RE = re.compile(r'####\s*(-?\d+(?:\.\d+)?)')

def parse_final_answer(content):
    m = _ANS_RE.search(content)
    if not m: return None
    raw = m.group(1)
    try:
        f = float(raw)
        return int(f) if abs(f - round(f)) < 1e-9 else f
    except ValueError:
        return None


def cleanup_solution(content):
    """Keep only step lines + final marker; strip explanations."""
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    keep = []
    for l in lines:
        if l.startswith('####') or '=' in l or any(c.isdigit() for c in l):
            keep.append(l)
    return '\n'.join(keep)


def process_one(item, model):
    prob = item['problem']
    expected = item.get('answer_real', item.get('answer'))
    try:
        expected_int = int(expected)
    except (TypeError, ValueError):
        return item, "no_int_answer", None

    content, tokens, dt, err = call_teacher(prob, model)
    if err is not None:
        return item, f"err:{err[:60]}", None

    teacher_ans = parse_final_answer(content)
    if teacher_ans is None:
        return item, "no_parsed_answer", None
    if int(teacher_ans) != expected_int:
        return item, f"mismatch:t={teacher_ans}_e={expected_int}", None

    new_item = dict(item)
    new_item['solution_cot'] = cleanup_solution(content)
    new_item['cot_tokens'] = tokens
    new_item['cot_seconds'] = round(dt, 2)
    new_item['cot_teacher'] = model
    return new_item, "ok", content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    items = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    if args.limit > 0:
        items = items[:args.limit]

    out_path = Path(args.output)
    done = set()
    if args.resume and out_path.exists():
        for l in out_path.read_text().splitlines():
            if not l.strip(): continue
            try:
                d = json.loads(l)
                done.add(d['problem'])
            except: pass
        print(f"Resume: {len(done)} pairs already done")

    todo = [it for it in items if it['problem'] not in done]
    print(f"Generating CoT for {len(todo)} problems via {args.model}, {args.workers} parallel")

    out_f = open(out_path, 'a' if args.resume else 'w')
    stats = {'ok':0,'mismatch':0,'no_parsed':0,'err':0,'no_int':0}
    total_tokens = 0
    t_start = time.time()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, it, args.model): it for it in todo}
        for i, f in enumerate(cf.as_completed(futures), 1):
            new_item, status, _ = f.result()
            if status == 'ok':
                stats['ok'] += 1
                total_tokens += new_item.get('cot_tokens', 0)
                out_f.write(json.dumps(new_item) + '\n')
                out_f.flush()
            elif status.startswith('mismatch'):
                stats['mismatch'] += 1
            elif status == 'no_parsed_answer':
                stats['no_parsed'] += 1
            elif status == 'no_int_answer':
                stats['no_int'] += 1
            else:
                stats['err'] += 1
            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1)
                eta = (len(todo) - i) / max(rate, 0.01)
                print(f"  [{i}/{len(todo)}] ok={stats['ok']} mismatch={stats['mismatch']} "
                      f"no_parsed={stats['no_parsed']} err={stats['err']} "
                      f"| {rate:.2f}/s eta={eta/60:.1f}m | tok={total_tokens}",
                      flush=True)
    out_f.close()
    print(f"\nDone in {(time.time()-t_start)/60:.1f}m. Stats: {stats}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
