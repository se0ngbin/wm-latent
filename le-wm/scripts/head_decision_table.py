"""Tables + paired exact McNemar tests (collisions) for the head-decision runs (K=15, crossing goals, 200 episodes).
Run: python head_decision_table.py"""
import re, os
import numpy as np
from scipy.stats import binomtest
L = "/home/seongbin/latent/run_logs"
CONDS = ["red", "purple", "fullrot90"]
VARIANTS = {"uniform h": ("{h}", "minh"), "predicted-latent h": ("{h}imag", "minh"), "per-step norm": ("{h}", "safe_task_norm")}
MODELS = {"jac s3072": "jacobian", "jac s1": "jacobian_s1", "jac s100": "jacobian_s100",
          "3f": "jacpull_linf_enc", "3g": "jacpull_linf_enc_linhead"}


def arr(s, key):
    m = list(re.finditer(rf"'{key}': array\(\[(.*?)\]", s, re.S))
    return np.array([x == "True" for x in re.findall(r"True|False", m[-1].group(1))]) if m else None


def load(tag, h, rule, c):
    f = f"{L}/softcem_{tag}_{h}_gr_w1000_k15_cross_ccem{rule}_{c}.log"
    if not os.path.exists(f): return None
    s = open(f).read()
    col, suc = arr(s, "episode_collisions"), arr(s, "episode_successes")
    return None if col is None or suc is None else (col, suc)


def mcnemar(a, b):   # bool collision arrays; counts a-only, b-only, exact two-sided p
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0)


def pair(a, b):
    return "—" if a is None or b is None or len(a[0]) != len(b[0]) else "%d/%d p=%.3f" % mcnemar(a[0], b[0])


R = {}
for vn, (hp, rule) in VARIANTS.items():
    print(f"\n## {vn}: collision % (success %)")
    for mn, tag in MODELS.items():
        for h in ["lin", "gp"]:
            row = []
            for c in CONDS:
                r = load(tag, hp.format(h=h), rule, c); R[vn, mn, h, c] = r
                row.append("—" if r is None else f"{100*r[0].mean():.1f} ({100*r[1].mean():.1f}) n={len(r[0])}")
            print(f"{mn:10s} {'linear' if h == 'lin' else 'MLP':6s} | " + " | ".join(row))
    print(f"\n## {vn}: paired collisions, linear vs MLP (linear-only / MLP-only, p)")
    for mn in MODELS:
        print(f"{mn:10s} | " + " | ".join(pair(R[vn, mn, "lin", c], R[vn, mn, "gp", c]) for c in CONDS))
    print(f"\n## {vn}: paired collisions, 3g vs 3f (3g-only / 3f-only, p)")
    for h in ["lin", "gp"]:
        print(f"h {h:4s} | " + " | ".join(pair(R[vn, "3g", h, c], R[vn, "3f", h, c]) for c in CONDS))
