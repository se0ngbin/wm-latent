"""Does jacobian reg make safe/unsafe states more linearly separable in latent space?

Labels dubins frames by distance to nearest obstacle edge (unsafe = near/inside,
safe = well clear), encodes them with baseline vs jacobian, and measures how well a
LINEAR probe recovers safety from the latent — a proxy for how cleanly a latent CBF
margin could be defined. Reports test AUC + accuracy (classification) and R²
(regressing the continuous edge-distance).
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
import stable_worldmodel as swm
from utils import get_img_preprocessor

DEV, IMG = "cuda:0", 224
OBS = np.array([(0.25, 0.65, 0.5), (0.25, -0.65, 0.5)])
NEAR, FAR = 0.10, 0.40          # unsafe if edge_dist<NEAR, safe if >FAR (drop middle)
N_SAMPLE = 12000                # frames to scan for labels
import os as _os
MODELS = {"baseline": "sigreg_only_dubins", "jacobian_w1": "jacobian_w1_dubins"}
if _os.path.exists("/data/seongbin/lewm/checkpoints/lewm_dubins_predanchor/weights_epoch_50.pt"):
    MODELS["pred_anchor"] = "lewm_dubins_predanchor"

def edge_dist(pos):
    return np.min([np.linalg.norm(pos - OBS[k, :2], axis=1) - OBS[k, 2] for k in range(2)], axis=0)

def encode(m, x):
    return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def main():
    ds = swm.data.load_dataset("dubins_expert.h5", transform=None, cache_dir=os.environ["LOCAL_DATASET_DIR"],
                               num_steps=1, frameskip=1, keys_to_load=["pixels", "state"], keys_to_cache=["state"])
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), N_SAMPLE, replace=False)
    st = np.stack([np.asarray(ds[int(i)]["state"][0]) for i in idx])
    ed = edge_dist(st[:, :2])
    unsafe = np.where(ed < NEAR)[0]; safe = np.where(ed > FAR)[0]
    n = min(len(unsafe), len(safe), 1500)              # balance classes
    sel = np.concatenate([rng.choice(unsafe, n, replace=False), rng.choice(safe, n, replace=False)])
    rng.shuffle(sel)
    frames = torch.stack([ds[int(idx[i])]["pixels"][0] for i in sel]).float()
    y = (ed[sel] < NEAR).astype(int); ed_sel = ed[sel]
    print(f"n_unsafe={len(unsafe)} n_safe={len(safe)} -> balanced set {2*n} (near<{NEAR}, far>{FAR})")

    for label, ckpt in MODELS.items():
        m = swm.wm.utils.load_pretrained(f"{ckpt}/weights_epoch_50.pt").to(DEV).eval()
        with torch.no_grad():
            Z = torch.cat([encode(m, frames[i:i+64].to(DEV)).cpu() for i in range(0, len(frames), 64)]).numpy()
        Ztr, Zte, ytr, yte, etr, ete = train_test_split(Z, y, ed_sel, test_size=0.3, random_state=0, stratify=y)
        sc = StandardScaler().fit(Ztr); Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
        p = clf.predict_proba(Zte)[:, 1]
        auc = roc_auc_score(yte, p); acc = accuracy_score(yte, p > 0.5)
        reg = Ridge(alpha=1.0).fit(Ztr, etr); r2 = r2_score(ete, reg.predict(Zte))
        print(f"{label:14s}  safe/unsafe AUC={auc:.3f}  acc={acc:.3f}  |  edge-dist R^2={r2:.3f}")
        del m; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
