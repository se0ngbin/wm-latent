"""Aggregate KEY lines from the self-contained OOD margin runs into one matrix.

Reads 'KEY ...' lines emitted by ood_margin_jepa_static.py and ood_margin_dreamer.py
(both self-contained: train+eval in one process, one renderer, rk=0). Reports, per
(model, encoder, axis), sign-acc + AUC of the three quantities the doc table needs:
  in-dist  = M_og  on og
  zeroshot = M_og  on new
  oracle   = M_new on new
mean±std across seeds when >1.
"""
import glob, re
from collections import defaultdict
import numpy as np

LOGS = "/home/seongbin/latent/run_logs"
PAT = re.compile(
    r"KEY (\w+) ([\w+-]+) axis=(\w+) rk=(\d+) seed=(\d+) "
    r"ogog_acc=([\d.]+) ogog_auc=([\d.]+) "
    r"zs_acc=([\d.]+) zs_auc=([\d.]+) or_acc=([\d.]+) or_auc=([\d.]+)")
ORDER = {"baseline": 0, "jacobian": 1, "jacobian+pull": 2, "enc-lip": 1, "invariance": 2}
AXORD = {"color": 0, "shape": 1, "rotate": 2}


def main(files=None):
    rows = defaultdict(lambda: defaultdict(list))
    seen = set()
    files = files or (glob.glob(f"{LOGS}/ood_gp_jepa_*.log") +
                      glob.glob(f"{LOGS}/ood_gp_dreamer_*.log"))
    for f in files:
        for line in open(f):
            m = PAT.search(line)
            if not m:
                continue
            model, enc, axis, rk, seed = m.group(1,2,3,4,5)
            if (model,enc,axis,rk,seed) in seen:
                continue
            seen.add((model,enc,axis,rk,seed))
            g = list(map(float, m.group(6,7,8,9,10,11)))
            k = (model, enc, axis)
            for name, val in zip(("ind_a","ind_u","zs_a","zs_u","or_a","or_u"), g):
                rows[k][name].append(val)

    def ms(v):
        return f"{np.mean(v):.3f}" + (f"±{np.std(v):.2f}" if len(v) > 1 else "")

    hdr = (f"{'model':8s} {'encoder':13s} {'axis':7s} {'n':>2s} | "
           f"{'ind acc':>8s} {'ind auc':>8s} | {'zs acc':>8s} {'zs auc':>8s} | "
           f"{'or acc':>8s} {'or auc':>8s}")
    print(hdr); print("-" * len(hdr))
    for k in sorted(rows, key=lambda k: (k[0], AXORD.get(k[2],9), ORDER.get(k[1],9))):
        model, enc, axis = k; d = rows[k]; n = len(d["zs_a"])
        print(f"{model:8s} {enc:13s} {axis:7s} {n:>2d} | "
              f"{ms(d['ind_a']):>8s} {ms(d['ind_u']):>8s} | "
              f"{ms(d['zs_a']):>8s} {ms(d['zs_u']):>8s} | "
              f"{ms(d['or_a']):>8s} {ms(d['or_u']):>8s}")


if __name__ == "__main__":
    main()
