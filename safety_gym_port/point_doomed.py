"""Dynamics-coupling probe for SafetyPointGoal1: fraction of currently-safe states (margin>0) from which
NO escape policy (9 constant actions + brake + 3 evade) avoids every hazard for H=80 steps.
'doomed' is an UPPER bound on true unsafety (finite policy set) -> also usable as an MC value label.

States are injected directly (Point = slide x/y + hinge z; slide axes are ROTATED at reset, so map
through d.xaxis): margin~U[0,.5] near a random hazard, speed~U[0,vmax], heading~velocity (ALIGN=1).

usage (sg venv, from safety-gymnasium/):
  MUJOCO_GL=egl .venv/bin/python ../safety_gym_port/point_doomed.py Point <n_layouts> <out.npy> <states_per_layout>
2026-10-22, 8 layouts x 1000: doomed band grows with approach speed to ~0.24 past the hazard edge at
top speed (= measured braking distance 0.238; hazard r=0.2); Car brakes in 0.055 -> ~no band.
"""
import sys, numpy as np, mujoco, safety_gymnasium as sg
AGENT = sys.argv[1]; NSEEDS = int(sys.argv[2]); H = 80; FS = 10
env = sg.make(f"Safety{AGENT}Goal1-v0"); 
rng = np.random.default_rng(0)

def setup(seed):
    env.reset(seed=seed); t = env.unwrapped.task
    return t, t.model, t.data, np.array([h[:2] for h in t.hazards.pos]), float(t.hazards.size)

def margin(t, H_, r):
    return float((np.linalg.norm(H_ - t.agent.pos[:2], axis=1) - r).min())

def heading(t):
    R = np.asarray(t.agent.mat).reshape(3, 3); return np.arctan2(R[1, 0], R[0, 0])

def vel(t):
    return np.asarray(t.agent.vel)[:2] if hasattr(t.agent, 'vel') else None

def step(m, d, a):
    d.ctrl[:] = np.clip(a, -1, 1)
    for _ in range(FS): mujoco.mj_step(m, d)

FIELDS = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "qacc_warmstart", "ctrl")
def get_state(d): return (d.time,) + tuple(getattr(d, f).copy() for f in FIELDS)
def set_state(m, d, s):
    d.time = s[0]
    for f, v in zip(FIELDS, s[1:]): getattr(d, f)[:] = v
    mujoco.mj_forward(m, d)

def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi

ROT_SIGN = 1.0
def steer_thrust(t, want):  # Point: point heading along +-want, thrust along want
    th = heading(t); ang = np.arctan2(want[1], want[0])
    e_f, e_b = wrap(ang - th), wrap(ang + np.pi - th)
    if abs(e_f) <= abs(e_b): return np.array([1.0 if abs(e_f) < 0.6 else 0.0, ROT_SIGN * 3 * e_f])
    return np.array([-1.0 if abs(e_b) < 0.6 else 0.0, ROT_SIGN * 3 * e_b])

def policies(H_):
    P = [lambda t, a=np.array([u, w]): a for u in (-1, 0, 1) for w in (-1, 0, 1)]
    if AGENT == "Point":
        v = lambda t: t.agent.pos[:2] - t._prev
        P.append(lambda t: steer_thrust(t, -v(t)) if np.linalg.norm(v(t)) > 1e-3 else np.zeros(2))  # brake
        def away(t, rot=0):
            dvec = t.agent.pos[:2] - H_[np.argmin(np.linalg.norm(H_ - t.agent.pos[:2], axis=1))]
            dvec = dvec / (np.linalg.norm(dvec) + 1e-9)
            if rot: dvec = np.array([-dvec[1], dvec[0]]) * rot
            return steer_thrust(t, dvec)
        P += [lambda t: away(t), lambda t: away(t, 1), lambda t: away(t, -1)]
    return P

def escapable(t, m, d, snap, H_, r):
    for pol in policies(H_):
        set_state(m, d, snap)
        t._prev = t.agent.pos[:2].copy(); ok = True
        for _ in range(H):
            a = pol(t); t._prev = t.agent.pos[:2].copy(); step(m, d, a)
            if margin(t, H_, r) < 0: ok = False; break
        if ok: return True
    return False

rows = []
import os; ALIGN = int(os.environ.get('ALIGN', 1))
_t, _m, *_ = setup(0)
jx, jy, jz = (_m.joint(n) for n in ("x", "y", "z"))
for seed in range(NSEEDS):
    t, m, d, H_, r = setup(seed)
    dt = m.opt.timestep * FS; vmax = 0.030 / dt   # top speed measured earlier: 0.030 per env step
    base = get_state(d)
    for _ in range(int(sys.argv[4])):
        set_state(m, d, base)
        h = H_[rng.integers(len(H_))]; ang = rng.uniform(0, 2*np.pi); want = h + (r + rng.uniform(0, .5)) * np.array([np.cos(ang), np.sin(ang)])
        A = np.stack([d.xaxis[jx.id][:2], d.xaxis[jy.id][:2]], 1)   # slide axes in world frame (agent body is rotated at reset)
        dq = np.linalg.solve(A, want - t.agent.pos[:2]); d.qpos[jx.qposadr[0]] += dq[0]; d.qpos[jy.qposadr[0]] += dq[1]
        sp = rng.uniform(0, vmax); va = rng.uniform(0, 2*np.pi)
        mujoco.mj_forward(m, d)   # heading ~ velocity dir (thrust is along heading); ALIGN=0 -> independent
        hd = va + rng.normal(0, .3) + (np.pi if rng.random() < .5 else 0) if ALIGN else rng.uniform(-np.pi, np.pi)
        d.qpos[jz.qposadr[0]] += wrap(hd - heading(t))
        d.qvel[jx.dofadr[0]], d.qvel[jy.dofadr[0]] = np.linalg.solve(A, sp*np.array([np.cos(va), np.sin(va)])); d.qvel[jz.dofadr[0]] = 0
        mujoco.mj_forward(m, d)
        assert np.allclose(t.agent.pos[:2], want, atol=1e-6) and np.allclose(t.agent.vel[:2], sp*np.array([np.cos(va), np.sin(va)]), atol=1e-6)
        mg = margin(t, H_, r)
        if mg < 0: continue
        nh = H_[np.argmin(np.linalg.norm(H_ - t.agent.pos[:2], axis=1))]; u = (nh - t.agent.pos[:2]); u /= np.linalg.norm(u)
        v_app = (sp*np.cos(va)*u[0] + sp*np.sin(va)*u[1]) * dt   # approach speed toward nearest hazard, per env step
        rows.append((mg, sp*dt, v_app, not escapable(t, m, d, get_state(d), H_, r)))
R = np.array(rows, float); np.save(sys.argv[3], R); mg, spd, vap, dm = R.T
print(f"{AGENT}: n={len(R)} margin~U[0,.5] speed~U[0,vmax]  doomed={dm.mean():.3f}  max margin among doomed={mg[dm==1].max() if dm.any() else 0:.3f}")
print("doomed fraction, rows=margin bin, cols=approach speed toward nearest hazard (per step, vmax=.030)")
vb = [-1, 0, .01, .02, .031]; mb = [0, .05, .1, .15, .2, .3, .5]
print("margin\\v_app " + " ".join(f"{'<0' if lo<0 else f'{lo:.2f}-{hi:.2f}':>10s}" for lo, hi in zip(vb, vb[1:])))
for a, b in zip(mb, mb[1:]):
    cells = []
    for lo, hi in zip(vb, vb[1:]):
        s_ = (mg>=a)&(mg<b)&(vap>=lo)&(vap<hi); cells.append(f"{dm[s_].mean():.2f}(n{s_.sum()})" if s_.any() else "-")
    print(f"[{a:.2f},{b:.2f})  " + " ".join(f"{c:>10s}" for c in cells))
