"""Markdown tables from /data/seongbin/lewm/support_alias/<env>.json (support_alias.py). Usage: tables.py dubins|cartpole|point"""
import json, sys

env = sys.argv[1]
res = json.load(open(f"/data/seongbin/lewm/support_alias/{env}.json"))
models = list(dict.fromkeys(k.split("|")[0] for k in res))
shifts = [s for s in dict.fromkeys(k.split("|")[1] for k in res) if s != "layout"]
f = lambda v: "—" if v != v else f"{v:.3f}"


def table(title, cell, base=None):
    cols = ([base[0]] if base else []) + shifts
    print(f"{title}\n\n| model | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|")
    for m in models:
        cells = ([base[1](res[f"{m}|{shifts[0]}"])] if base else []) + [cell(res[f"{m}|{s}"]) for s in shifts]
        print(f"| {m} | " + " | ".join(cells) + " |")
    print()


if env == "point":
    print("Layout shift (base view, unseen layouts vs held-out frames of training layouts)\n\n"
          "| model | off support | distance ratio | head sign error | kNN class mismatch (in support) |\n|---|---|---|---|---|")
    for m in models:
        r = res[f"{m}|layout"]
        print(f"| {m} | {f(r['off'])} | {r['dist_ratio']:.2f} | {f(r['wrong'])} | {f(r['alias_sh'])} |")
    print()
table("Off-support fraction (kNN distance above the 99th percentile of native test frames)", lambda r: f(r["off"]))
table("Sign flips vs the native frame: inside support / off support", lambda r: f"{f(r['in_flip'])} / {f(r['off_flip'])}")
table("kNN-neighbourhood AUC (fraction of safe training neighbours as the score)",
      lambda r: f(r["knn_auc_sh"]), base=("native", lambda r: f(r["knn_auc_og"])))
table("Unsafe states whose nearest training latents are mostly safe",
      lambda r: f(r["alias_unsafe_sh"]), base=("native", lambda r: f(r["alias_unsafe_og"])))
