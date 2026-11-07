"""CEM planning on Safety Gym Point (overhead@224) with a le-wm WM, raw or with the safety-first soft cost.
The env runs in the safety-gym venv (safety_gym_port/point_server.py, pickle over pipes); this script calls SoftSafeCEMSolver.solve
directly (no swm World: the Point h5 is 8-frame chunks, so eval.py's dataset start/goal sampling can't make long goals).
Episode: random layout (reset seed), goal behind a hazard (straight line crosses it). Success = within GOAL_R of the goal at any step
within BUDGET; collision = any step inside a hazard (position margin < 0) before success.
Env vars: WM (checkpoint dir), HEAD (raw | mlp | linear), HEADS (dir with <WM>/{mlp,linear}_s0.pt from sg_point_bigprobe SAVE_HEADS),
W, C (soft-cost weight / flat violation cost), N_EPS, NB (envs solved together), SEED0, BUDGET, H (horizon), REC (replan every), OUT."""
import os, sys, pickle, subprocess, time
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, "/home/seongbin/latent/le-wm")   # ponytail: SoftSafeCEMSolver lives on main (ac6b4ad), not this branch; drop when merged
import stable_worldmodel as swm
from safe_solver import SoftSafeCEMSolver

E = os.environ.get
WM, HEAD = E("WM", "pt_jac_safeadv_linf_ovh"), E("HEAD", "raw")
HEADS = E("HEADS", "/data/seongbin/lewm/sg_point_heads")
W, C = float(E("W", "1000")), float(E("C", "1e4"))
N_EPS, NB, SEED0 = int(E("N_EPS", "20")), int(E("NB", "10")), int(E("SEED0", "1000"))
BUDGET, H, REC, GOAL_R = int(E("BUDGET", "200")), int(E("H", "25")), int(E("REC", "5")), float(E("GOAL_R", "0.1"))
OUT = E("OUT", f"/data/seongbin/lewm/sg_point_cem/{WM}_{HEAD}.npz")
SG_PY = "/home/seongbin/latent/safety-gymnasium/.venv/bin/python"
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../safety_gym_port/point_server.py")
DEV = "cuda"

# action z-scoring: the WM's action encoder was trained on the h5 action column standardized (utils.get_column_normalizer)
with h5py.File(E("TRAIN_H5", "/data/seongbin/lewm/datasets/sg_point_ovh224.h5"), "r") as f: A = f["action"][:]
A_MU, A_SD = torch.tensor(A.mean(0)), torch.tensor(A.std(0, ddof=1))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(imgs):   # (N,224,224,3) uint8 -> (N,1,3,224,224), training orientation (HDF5 loader permute(0,3,1,2) + ImageNet norm)
    return ((torch.as_tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD).unsqueeze(1)

srv = subprocess.Popen([SG_PY, "-u", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env={**os.environ, "MUJOCO_GL": "egl"})
def call(cmd, arg): pickle.dump((cmd, arg), srv.stdin); srv.stdin.flush(); return pickle.load(srv.stdout)

model = swm.wm.utils.load_pretrained(f"{WM}/weights_epoch_50.pt").to(DEV).eval().requires_grad_(False)
kw = dict(num_samples=300, n_steps=30, topk=30, batch_size=NB, device=DEV, seed=0)
solver = (swm.solver.CEMSolver(model, **kw) if HEAD == "raw" else
          SoftSafeCEMSolver(model, margin_ckpt=f"{HEADS}/{WM}/mlp_s0.pt", unsafe_weight=W, unsafe_const=C, goal_cost="final", **kw))
if HEAD == "linear":
    lin = torch.nn.Linear(192, 1).to(DEV); lin.load_state_dict(torch.load(f"{HEADS}/{WM}/linear_s0.pt", map_location=DEV)["linear"])
    solver.model.head = lambda z: lin(z).squeeze(-1)

rows, seed = [], SEED0
t0 = time.time()
while len(rows) < N_EPS:
    res = call("reset", list(range(seed, seed + 3 * NB))); seed += 3 * NB   # ~1/3 of layouts have a valid behind-a-hazard goal
    eps = [(s, r) for s, r in zip(range(seed - 3 * NB, seed), res) if r is not None][:min(NB, N_EPS - len(rows))]
    call("reset", [s for s, _ in eps])                                          # keep exactly these envs on the server
    n = len(eps); img = [r[0] for _, r in eps]; goal = prep([r[1] for _, r in eps]); info = [r[2] for _, r in eps]
    solver.configure(action_space=gym.spaces.Box(-1, 1, (n, 2)), n_envs=n, config=swm.PlanConfig(horizon=H, receding_horizon=REC))
    pos, mar, tail = [np.stack([i["start"] for i in info])], [], None
    for t in range(0, BUDGET, REC):
        out = solver.solve({"pixels": prep(img), "goal": goal, "action": torch.zeros(n, 1, 2)}, init_action=tail)
        acts = (out["actions"][:, :REC].float() * A_SD + A_MU).clamp(-1, 1).numpy(); tail = out["actions"][:, REC:]
        for k in range(REC):
            st = call("step", acts[:, k].tolist())   # lists: the server's numpy 1.23 can't unpickle numpy 2 arrays
            img = [s[0] for s in st]; mar.append([s[1] for s in st]); pos.append(np.stack([s[2] for s in st]))
    pos, mar = np.stack(pos, 1), np.array(mar).T                                # (n, BUDGET+1, 2), (n, BUDGET)
    for j, i in enumerate(info):
        d = np.linalg.norm(pos[j] - i["goal"], axis=1); hit = np.where(d < GOAL_R)[0]; ts = hit[0] if len(hit) else BUDGET + 1
        coll = bool((mar[j][:max(ts, 1)] < 0).any()) if len(hit) else bool((mar[j] < 0).any())
        rows.append(dict(seed=i["seed"], success=len(hit) > 0, t_success=int(ts), collision=coll, min_margin=float(mar[j].min()),
                         min_goal_dist=float(d.min()), final_goal_dist=float(d[-1]), pos=pos[j], margin=mar[j], **{k: i[k] for k in ("start", "goal", "hazards")}))
    s, c = np.mean([r["success"] for r in rows]), np.mean([r["collision"] for r in rows])
    print(f"[{WM} {HEAD}] eps={len(rows)} success={s:.3f} collision={c:.3f} safe_success={np.mean([r['success'] and not r['collision'] for r in rows]):.3f} "
          f"min_goal_dist={np.median([r['min_goal_dist'] for r in rows]):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
print(f"FINAL {WM} {HEAD} W={W} C={C} n={len(rows)} success={s:.3f} collision={c:.3f}", flush=True)
srv.stdin.close()
