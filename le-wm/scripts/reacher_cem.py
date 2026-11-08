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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # this branch's safe_solver (has CCEM)
import stable_worldmodel as swm
from safe_solver import SoftSafeCEMSolver

E = os.environ.get
WM, EP, HEAD = E("WM", "lewm_reacher"), E("EP", "50"), E("HEAD", "raw")
HEADS = E("HEADS", "/data/seongbin/lewm/reacher_heads")
W, C = float(E("W", "1000")), float(E("C", "1e4"))
CCEM = E("CCEM", "")   # e.g. minh: constrained elite rule (as Dubins CCEM); empty = legacy safety-first soft cost
MODE = E("MODE", "data")   # data: LeWM protocol pairs that cross the box (default); sls: SLS-squared's 2.7 rad sweep (fails: goal cost flat)
N_EPS, H, REC, FS = int(E("N_EPS", "50")), int(E("H", "5")), int(E("REC", "5")), 5
BUDGET = int(E("BUDGET", "50" if MODE == "data" else "200"))
NITER = int(E("NITER", "10"))   # LeWM paper: 30 CEM iterations for PushT, 10 for the other environments
OUT = E("OUT", f"/data/seongbin/lewm/reacher_cem/{WM}_{HEAD}_{MODE}.npz")
DEV = "cuda"
START_C, START_HW, GOAL = np.array([3.1, -2.25]), np.array([0.05, 0.1]), np.array([0.37, -2.09])
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45

with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f: A = f["action"][:]
A = A[~np.isnan(A).any(1)]   # each episode's last frame has a NaN action; eval.py drops these before fitting its scaler
A_MU, A_SD = torch.tensor(A.mean(0), dtype=torch.float32), torch.tensor(A.std(0), dtype=torch.float32)   # StandardScaler, as eval.py
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(imgs): return ((torch.as_tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD).unsqueeze(1)

model = swm.wm.utils.load_pretrained(f"{WM}/weights_epoch_{EP}.pt").to(DEV).eval().requires_grad_(False)
kw = dict(num_samples=300, n_steps=NITER, topk=30, batch_size=int(E("NB", "10")), device=DEV, seed=int(E("CEM_SEED", "42")))   # eval.py: solver seed = cfg.seed = 42   # NB envs per CEM batch (GPU memory)
solver = (swm.solver.CEMSolver(model, **kw) if HEAD == "raw" else
          SoftSafeCEMSolver(model, margin_ckpt=f"{HEADS}/{WM}/mlp_s0.pt", unsafe_weight=W, unsafe_const=C, goal_cost="final",
                            ccem=bool(CCEM), ccem_fallback=CCEM or "count", **kw))
if HEAD == "linear":
    lin = torch.nn.Linear(192, 1).to(DEV); lin.load_state_dict(torch.load(f"{HEADS}/{WM}/linear_s0.pt", map_location=DEV)["linear"])
    solver.model.head = lambda z: lin(z).squeeze(-1)
solver.configure(action_space=gym.spaces.Box(-1, 1, (N_EPS, 2)), n_envs=N_EPS, config=swm.PlanConfig(horizon=H, receding_horizon=REC, action_block=FS))

rng = np.random.default_rng(0)
envs = [gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped for _ in range(N_EPS)]
if MODE == "lewm":
    # Exactly eval.py's selection (config/eval/reacher.yaml): every start with step_idx <= ep_len - 25 - 1 over the whole dataset,
    # rng(seed=42).choice(len - 1, num_eval, replace=False), sorted; goal = the frame 25 steps later. Only the box check is added.
    GOFF = 25
    with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
        ep, st, el, Q, V = f["ep_idx"][:], f["step_idx"][:], f["ep_len"][:], f["qpos"][:], f["qvel"][:]
        valid = np.nonzero(st <= (el - GOFF - 1)[ep])[0]
        sel = np.sort(valid[np.random.default_rng(int(E("PAIR_SEED", "42"))).choice(len(valid) - 1, size=N_EPS, replace=False)])
        goal = prep([f["pixels"][i + GOFF] for i in sel]); GOALS = Q[sel + GOFF]
        cross = np.array([f["failures"][i:i + GOFF + 1].any() for i in sel])
    print(f"LeWM pairs: {len(valid)} valid starts; {cross.sum()}/{N_EPS} selected pairs have a recorded path that enters the box", flush=True)
    starts = Q[sel]
    for e, i in zip(envs, sel): e.reset(seed=0); e.set_state(Q[i], V[i])
elif MODE == "data":
    # LeWM eval protocol (config/eval/reacher.yaml): start = dataset state (qpos, qvel), goal = the dataset frame GOFF steps later.
    # Held-out episodes (>= 9000, unseen by the margin functions). Kept: start and goal outside the box, but the dataset's own path
    # between them enters it, so the recorded route is unsafe and a safe plan has to deviate.
    GOFF = 25
    with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
        fl, ep, Q, V = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:]
        cs = np.r_[0, np.cumsum(fl)]
        t = np.arange(len(fl) - GOFF)
        ok = (ep[t] >= 9000) & (ep[t] == ep[t + GOFF]) & (fl[t] == 0) & (fl[t + GOFF] == 0) & (cs[t + GOFF + 1] - cs[t] > 0)
        cand = t[ok]; print(f"candidate pairs {len(cand)} (held-out episodes, safe endpoints, dataset path crosses the box)", flush=True)
        sel = np.sort(cand[np.random.default_rng(int(E("PAIR_SEED", "42"))).choice(len(cand) - 1, size=N_EPS, replace=False)])   # as eval.py
        goal = prep([f["pixels"][i + GOFF] for i in sel]); GOALS = Q[sel + GOFF]
    starts = Q[sel]
    for e, i in zip(envs, sel): e.reset(seed=0); e.set_state(Q[i], V[i])
else:
    starts = rng.uniform(START_C - START_HW, START_C + START_HW, (N_EPS, 2)); GOALS = np.tile(GOAL, (N_EPS, 1))
    goal_img = []
    for e, s in zip(envs, starts):
        e.reset(seed=0); e.set_state(GOAL, np.zeros(2)); goal_img.append(e.render()); e.set_state(s, np.zeros(2))
    goal = prep(goal_img)
# OOD appearance at planning time; start AND goal frames rendered under it. FLOOR=a: floor = (1-a) default checker + a brown
# (as reacher_ood FLOOR_SWEEP); LIGHT=i: light intensity (default 0.7); ARM=1: purple arm (reacher_ood arm_color)
VAR = {}
if E("FLOOR"): a = float(E("FLOOR")); VAR["floor.color"] = (1 - a) * np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]) + a * np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
if E("LIGHT"): VAR["light.intensity"] = np.array([float(E("LIGHT"))])
if E("ARM"): VAR["agent.color"] = np.array([0.55, 0.2, 0.75])
SHIFT = E("SHIFT", "")   # sim-to-real shift from reacher_shift.py (camera/light/sensor/dynamics); goal frame rendered under it too
import reacher_shift as RS
R = lambda e, j: RS.corrupt(e.render(), SHIFT, j) if SHIFT else e.render()
if VAR or SHIFT:
    assert MODE in ("data", "lewm")
    gi = []
    for j, (e, i) in enumerate(zip(envs, sel)):
        e.reset(seed=0, options={"variation_values": VAR} if VAR else None); RS.apply_physics(e, SHIFT)
        e.set_state(Q[i + GOFF], V[i + GOFF]); gi.append(R(e, j)); e.set_state(Q[i], V[i])
    goal = prep(gi)
img = [R(e, j) for j, e in enumerate(envs)]
prev = np.zeros((N_EPS, 2)); arng = np.random.default_rng(7)   # delay1 / act_noise state
qs = [[s.copy()] for s in starts]
t0, tail = time.time(), None
for t in range(0, BUDGET, REC * FS):
    out = solver.solve({"pixels": prep(img), "goal": goal, "action": torch.zeros(N_EPS, 1, 2 * FS)}, init_action=tail)
    tail = out["actions"][:, REC:]
    plan = out["actions"][:, :REC].float().reshape(N_EPS, REC * FS, 2) * A_SD + A_MU   # policy.py: (blocks, 2*FS) -> (steps, 2), inverse z-score
    for k in range(REC * FS):
        for j, e in enumerate(envs):
            a = np.clip(plan[j, k].numpy(), -1, 1)
            if SHIFT == "delay1": a, prev[j] = prev[j].copy(), a
            if SHIFT == "act_noise": a = np.clip(a + arng.normal(0, 0.2, 2), -1, 1)
            e.step(a); qs[j].append(np.asarray(e.env.physics.data.qpos[:2]).copy())
    img = [R(e, j) for j, e in enumerate(envs)]
    print(f"[{WM} {HEAD}] step {t + REC * FS}/{BUDGET} ({(time.time() - t0) / 60:.1f} min)", flush=True)
qs = np.array(qs)                                                                    # (N, BUDGET+1, 2)
hit = np.all(np.abs(qs - GOALS[:, None]) < 0.05, -1); ts = np.where(hit.any(1), hit.argmax(1), BUDGET + 1)
box = np.array([[in_box(q) for q in traj] for traj in qs])
coll = np.array([box[j, :ts[j]].any() if ts[j] <= BUDGET else box[j].any() for j in range(N_EPS)])
succ = ts <= BUDGET
print(f"FINAL {WM} {HEAD} W={W} C={C} n={N_EPS} success={succ.mean():.3f} collision={coll.mean():.3f} safe_success={(succ & ~coll).mean():.3f} "
      f"min_goal_dist(median)={np.median(np.abs(qs - GOALS[:, None]).max(-1).min(1)):.3f}", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, q=qs, start=starts, success=succ, collision=coll, t_success=ts, in_box=box)
