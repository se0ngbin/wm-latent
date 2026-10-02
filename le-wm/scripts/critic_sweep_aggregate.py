"""Aggregate RESULT lines from critic_sweep_results.txt into mean±std over seeds.
Prints a per-model table: in-dist, and per shift (zero-shot -> retrained), acc/auc."""
import sys, re
from collections import defaultdict
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/home/seongbin/latent/run_logs/critic_sweep_results.txt"
acc = defaultdict(list); auc = defaultdict(list)
for ln in open(path):
    m = re.match(r"RESULT (\S+) (\S+) (\S+) (\S+) ([0-9.]+) ([0-9.]+)", ln.strip())
    if not m: continue
    model, col, cond, seed, a, u = m.groups()
    acc[(model, col, cond)].append(float(a)); auc[(model, col, cond)].append(float(u))

def ms(vals):
    v = np.array(vals);
    return f"{v.mean():.3f}±{v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"

def cell(model, col, cond):
    k = (model, col, cond)
    if k not in acc: return "  n/a  "
    return f"{ms(acc[k])} / {ms(auc[k])} (n={len(acc[k])})"

models = ["baseline", "jacobian", "jacpull", "dreamer"]
print("\n===== Critic sign-acc / AUC vs HJ V*  (mean±std over seeds) =====\n")
for model in models:
    print(f"## {model}")
    print(f"  in-dist (red)      : {cell(model,'indist','red')}")
    for cond in ["purple", "diamond", "rot90"]:
        print(f"  {cond:8s} zero-shot : {cell(model,'zeroshot',cond)}")
        print(f"  {cond:8s} retrained : {cell(model,'retrained',cond)}")
    print()

# Markdown table (zero-shot -> retrained per shift), acc±/auc±
print("\n===== Markdown (acc±std / auc±std; shift cells = zero-shot -> retrained) =====\n")
print("| model | in-dist (red) | color: zs -> retr | shape: zs -> retr | rotation: zs -> retr |")
print("|---|---|---|---|---|")
name = {"baseline":"JEPA baseline","jacobian":"JEPA jacobian","jacpull":"JEPA jac+pull","dreamer":"Dreamer baseline"}
for model in models:
    def pair(cond):
        return f"{cell(model,'zeroshot',cond).split(' (n')[0]} → {cell(model,'retrained',cond).split(' (n')[0]}"
    row = f"| {name[model]} | {cell(model,'indist','red').split(' (n')[0]} | {pair('purple')} | {pair('diamond')} | {pair('rot90')} |"
    print(row)
