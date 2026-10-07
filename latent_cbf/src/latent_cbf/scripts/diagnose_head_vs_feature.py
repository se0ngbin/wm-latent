"""Disentangle the color-robustness seam: the margin STUDY (projector feature,
fresh red-trained head) has jacpull surviving color (AUC ~.75); the DEPLOYED head
(get_feat feature, margin_heads.pt) inverts (~.42). Which factor causes it?

Same sampled states, each pipeline run NATIVELY (study renders 224 for the raw-JEPA
projector; deployed renders its own size for the adapter get_feat), then cross
feature x head into three cells scored zero-shot on color (M_red on purple):
  A  get_feat  + deployed head (margin_heads.pt)   -> the probe/deployed number
  B  get_feat  + fresh red-trained head            -> same feature, different head
  C  projector + fresh red-trained head            -> the margin-study number
B vs A isolates the HEAD WEIGHTS; C vs B isolates the FEATURE.
Usage: diagnose_head_vs_feature.py [seed]
"""
import sys, os
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm", "/home/seongbin/latent/le-wm/scripts"):
    if _p not in sys.path: sys.path.append(_p)
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import gym as old_gym  # noqa
import gymnasium as gym  # noqa
import stable_worldmodel as swm
import ood_margin_gp_jepa as S                       # study funcs + constants
from utils import get_img_preprocessor
from critic_ood_eval import render_cond as get_render, feats as jepa_feats, CONDS, DEV
from configs import DreamerConfig, Config
from latent_cbf.adapters import LEWMWorldModel
from latent_cbf.dubins.dubins_env import DubinsEnv

CK = "/data/seongbin/lewm/checkpoints"
MG = "/data/seongbin/dreamer/lewm"
WM = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
      "jacpull": "lewm_dubins_jacpull50"}
NTR = S.NTR


def score(head, Z, y):
    with torch.no_grad():
        h = head(Z.to(DEV)).view(-1).cpu().numpy()
    safe = (y == 0).astype(int)
    acc = float(((h >= 0) == (safe == 1)).mean())
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(safe, h)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = S.WORLD_BOUNDS
    st = np.stack([rng.uniform(x0, x1, S.N), rng.uniform(y0, y1, S.N),
                   rng.uniform(-np.pi, np.pi, S.N)], 1).astype(np.float32)
    y_circle = S.inside_obstacle(st[:, :2], "circle")

    ec = Config().environment
    obst = ec.get_obstacles_list()
    # confirm the two pipelines share obstacle geometry (else labels wouldn't pair)
    print(f"seed={seed} N={S.N} frac_unsafe(study)={y_circle.mean():.3f} "
          f"study_obst={S.OBSTACLES} probe_obst={obst} study_bounds={S.WORLD_BOUNDS} "
          f"probe_bounds={tuple(ec.world_bounds)}")

    for tag, w in WM.items():
        ck = f"{CK}/{w}/weights_epoch_50.pt"
        # ---- get_feat path (deployed adapter pipeline) ----
        denv = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                         max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                         obstacles=obst, goal_radius=ec.goal_radius,
                         collision_radius=ec.collision_radius, render_mode="rgb_array")
        cfg = DreamerConfig(); cfg.lewm_ckpt_path = ck
        wm = LEWMWorldModel(cfg, ck).to(DEV)
        sd = torch.load(f"{MG}/{w}/margin_heads.pt", map_location=DEV)
        wm.heads["margin_gp"].load_state_dict(sd["margin_gp"]); wm.eval()
        deployed = wm.heads["margin_gp"]
        Fg_red = jepa_feats(wm, get_render(denv, st, CONDS["red"])).cpu()
        Fg_pur = jepa_feats(wm, get_render(denv, st, CONDS["purple"])).cpu()
        del wm; torch.cuda.empty_cache()

        # ---- projector path (raw-JEPA study pipeline) ----
        senv = S.DubinsSwmEnv()
        tf = get_img_preprocessor(source="pixels", target="pixels", img_size=S.IMG)
        m = swm.wm.utils.load_pretrained(ck).to(DEV).eval()
        og, new = S.AXES["color"][0], S.AXES["color"][1]
        Fp_red = S.encode(m, S.render_cond(senv, st, og, tf))
        Fp_pur = S.encode(m, S.render_cond(senv, st, new, tf))
        del m; torch.cuda.empty_cache()

        # ---- fresh red-trained heads (identical study protocol) ----
        h_get_fresh = S.train_margin(Fg_red[:NTR], y_circle[:NTR], "gp", seed=seed)
        h_proj_fresh = S.train_margin(Fp_red[:NTR], y_circle[:NTR], "gp", seed=seed)

        yte = y_circle[NTR:]
        cells = {
            "A_getfeat_deployed": (deployed, Fg_red, Fg_pur),
            "B_getfeat_fresh":    (h_get_fresh, Fg_red, Fg_pur),
            "C_projector_fresh":  (h_proj_fresh, Fp_red, Fp_pur),
        }
        print(f"\n=== {tag} ({w}) color ===")
        for name, (head, Zr, Zp) in cells.items():
            ia, iu = score(head, Zr[NTR:], yte)          # in-dist (red head on red)
            za, zu = score(head, Zp[NTR:], yte)          # zero-shot (red head on purple)
            print(f"HEADFEAT {tag} {name} indist_acc={ia:.3f} indist_auc={iu:.3f} "
                  f"zs_acc={za:.3f} zs_auc={zu:.3f}")
    print("\nDIAG_DONE")


if __name__ == "__main__":
    main()
