"""Goal-marker nuisance test: the saved GP margin functions were trained on renders with the env's default goal marker.
Score them on fresh uniform states rendered (a) with that default marker and (b) with the marker moved to random goal
positions drawn from the expert dataset. A large AUC drop in (b) means the latent's safety readout depends on the goal
marker, a nuisance in planning frames. Frames in the training orientation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from safe_solver import _MarginHead

N = 3000
ENC = {"sigreg": "sigreg_only_dubins", "jac": "jacobian_w1_dubins", "jac+pull": "lewm_dubins_jacpull50",
       "adv_h": "lewm_dubins_safeadv_linf50", "jac+pull+adv_h": "lewm_dubins_jacpull_safeadv_linf50",
       "adv_λ": "lewm_dubins_lam_bdet50"}
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
rng = np.random.default_rng(7); x0, x1, y0, y1 = E.WORLD_BOUNDS
s = np.stack([rng.uniform(x0, x1, N), rng.uniform(y0, y1, N), rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
cond = E.AXES["color"][0]; y = E.inside_obstacle(s[:, :2], cond["shape"])
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    st = f["state"][:]
goals = st[rng.integers(0, len(st), N), :2]                       # goal positions seen in planning frames
env = E.DubinsSwmEnv()
default_goal = env.goal_position.copy()
F_fixed = E.render_cond(env, s, cond, tf_fixed)
frames = []
for i in range(N):                                                # same states, goal marker moved per frame
    env.goal_position = goals[i]
    frames.append(E.render_cond(env, s[i:i + 1], cond, tf_fixed))
env.goal_position = default_goal
F_moved = torch.cat(frames)
for name, ck in ENC.items():
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
    head = _MarginHead().to(E.DEV)
    head.load_state_dict(torch.load(f"/data/seongbin/dreamer/lewm/{ck}/margin_gp_fail_red_s0.pt", map_location=E.DEV)["margin_gp"])
    head.eval()
    _, a_fixed = E.score(head, E.encode(m, F_fixed), y)
    _, a_moved = E.score(head, E.encode(m, F_moved), y)
    print(f"GOALMARKER {name} AUC default-marker={a_fixed:.3f} moved-marker={a_moved:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
