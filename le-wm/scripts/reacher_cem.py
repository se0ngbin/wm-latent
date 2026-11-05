"""CEM on DMC Reacher with SLS-squared's task: start qpos ~ U([3.1, -2.25] +- [0.05, 0.1]) at rest, goal qpos [0.37, -2.09]
(SLS-squared/reacher/plan/generate_start_goal.py); invisible joint box unsafe iff wrap(q1) in [0, pi] and q2 in [-2.88, -2.45]
(obs_data_collect_new.py). The shoulder sweeps across the box's q1 range with the elbow 0.2-0.36 rad above it, so collisions come from
elbow lag when the shoulder accelerates. Success = every joint within 0.05 of the goal (the env's qpos_match rule) within BUDGET steps;
collision = any env step inside the box before success. Planning as LeWM's reacher eval: action_block 5, horizon 5 blocks, replan every
REC blocks, 300 samples x 30 iters, actions z-scored per raw action with the dataset's stats.
Env: WM, EP, HEAD (raw | mlp | linear), HEADS, W, C, N_EPS, BUDGET, H, REC, OUT."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, "/home/seongbin/latent/le-wm")   # ponytail: SoftSafeCEMSolver lives on main, not this branch; drop when merged
import stable_worldmodel as swm
from safe_solver import SoftSafeCEMSolver

E = os.environ.get
WM, EP, HEAD = E("WM", "lewm_reacher"), E("EP", "50"), E("HEAD", "raw")
HEADS = E("HEADS", "/data/seongbin/lewm/reacher_heads")
W, C = float(E("W", "1000")), float(E("C", "1e4"))
N_EPS, BUDGET, H, REC, FS = int(E("N_EPS", "50")), int(E("BUDGET", "200")), int(E("H", "5")), int(E("REC", "5")), 5
OUT = E("OUT", f"/data/seongbin/lewm/reacher_cem/{WM}_{HEAD}.npz")
DEV = "cuda"
START_C, START_HW, GOAL = np.array([3.1, -2.25]), np.array([0.05, 0.1]), np.array([0.37, -2.09])
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45

with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f: A = f["action"][:]
A = A[~np.isnan(A).any(1)]   # each episode's last frame has a NaN action; eval.py drops these before fitting its scaler
A_MU, A_SD = torch.tensor(A.mean(0), dtype=torch.float32), torch.tensor(A.std(0), dtype=torch.float32)   # StandardScaler, as eval.py
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(imgs): return ((torch.as_tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD).unsqueeze(1)

model = swm.wm.utils.load_pretrained(f"{WM}/weights_epoch_{EP}.pt").to(DEV).eval().requires_grad_(False)
kw = dict(num_samples=300, n_steps=30, topk=30, batch_size=int(E("NB", "10")), device=DEV, seed=0)   # NB envs per CEM batch (GPU memory)
solver = (swm.solver.CEMSolver(model, **kw) if HEAD == "raw" else
          SoftSafeCEMSolver(model, margin_ckpt=f"{HEADS}/{WM}/mlp_s0.pt", unsafe_weight=W, unsafe_const=C, goal_cost="final", **kw))
if HEAD == "linear":
    lin = torch.nn.Linear(192, 1).to(DEV); lin.load_state_dict(torch.load(f"{HEADS}/{WM}/linear_s0.pt", map_location=DEV)["linear"])
    solver.model.head = lambda z: lin(z).squeeze(-1)
solver.configure(action_space=gym.spaces.Box(-1, 1, (N_EPS, 2)), n_envs=N_EPS, config=swm.PlanConfig(horizon=H, receding_horizon=REC, action_block=FS))

rng = np.random.default_rng(0)
starts = rng.uniform(START_C - START_HW, START_C + START_HW, (N_EPS, 2))
envs = [gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped for _ in range(N_EPS)]
goal_img = []
for e, s in zip(envs, starts):
    e.reset(seed=0); e.set_state(GOAL, np.zeros(2)); goal_img.append(e.render()); e.set_state(s, np.zeros(2))
goal = prep(goal_img)
img = [e.render() for e in envs]
qs = [[s.copy()] for s in starts]
t0, tail = time.time(), None
for t in range(0, BUDGET, REC * FS):
    out = solver.solve({"pixels": prep(img), "goal": goal, "action": torch.zeros(N_EPS, 1, 2 * FS)}, init_action=tail)
    tail = out["actions"][:, REC:]
    plan = out["actions"][:, :REC].float().reshape(N_EPS, REC * FS, 2) * A_SD + A_MU   # policy.py: (blocks, 2*FS) -> (steps, 2), inverse z-score
    for k in range(REC * FS):
        for j, e in enumerate(envs):
            e.step(np.clip(plan[j, k].numpy(), -1, 1)); qs[j].append(np.asarray(e.env.physics.data.qpos[:2]).copy())
    img = [e.render() for e in envs]
    print(f"[{WM} {HEAD}] step {t + REC * FS}/{BUDGET} ({(time.time() - t0) / 60:.1f} min)", flush=True)
qs = np.array(qs)                                                                    # (N, BUDGET+1, 2)
hit = np.all(np.abs(qs - GOAL) < 0.05, -1); ts = np.where(hit.any(1), hit.argmax(1), BUDGET + 1)
box = np.array([[in_box(q) for q in traj] for traj in qs])
coll = np.array([box[j, :ts[j]].any() if ts[j] <= BUDGET else box[j].any() for j in range(N_EPS)])
succ = ts <= BUDGET
print(f"FINAL {WM} {HEAD} W={W} C={C} n={N_EPS} success={succ.mean():.3f} collision={coll.mean():.3f} safe_success={(succ & ~coll).mean():.3f} "
      f"min_goal_dist(median)={np.median(np.abs(qs - GOAL).max(-1).min(1)):.3f}", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, q=qs, start=starts, success=succ, collision=coll, t_success=ts, in_box=box)
