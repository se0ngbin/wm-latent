"""Goal-seeking controller for the Car (matches the dubins 'expert' collection: drive
toward the goal, crossing hazards, giving ~17% unsafe frames). Differential drive:
action=[left,right]; forward=[+,+], turn via wheel difference. + exploration noise."""
import numpy as np

def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def goal_seek_action(task, rng, forward=0.8, kturn=2.0, noise=0.35):
    x, y = float(task.agent.pos[0]), float(task.agent.pos[1])
    m = np.asarray(task.agent.mat).reshape(3, 3); th = np.arctan2(m[1, 0], m[0, 0])
    gx, gy = float(task.goal.pos[0]), float(task.goal.pos[1])
    e = _wrap(np.arctan2(gy - y, gx - x) - th)     # heading error to goal
    turn = np.clip(kturn * e, -1, 1)
    a = np.array([forward + turn, forward - turn], dtype=np.float64)   # left>right => turn +
    a += rng.normal(0, noise, size=2)               # exploration
    return np.clip(a, -1, 1)


def hazard_seek_action(task, rng, forward=0.7, kturn=2.0, noise=0.4):
    """Drive toward the NEAREST hazard center (generates unsafe transitions)."""
    x, y = float(task.agent.pos[0]), float(task.agent.pos[1])
    m = np.asarray(task.agent.mat).reshape(3, 3); th = np.arctan2(m[1, 0], m[0, 0])
    hs = np.asarray([h[:2] for h in task.hazards.pos])
    d = np.linalg.norm(hs - np.array([x, y]), axis=1); tgt = hs[d.argmax() if rng.random() < 0.0 else d.argmin()]
    e = _wrap(np.arctan2(tgt[1] - y, tgt[0] - x) - th)
    turn = np.clip(kturn * e, -1, 1)
    a = np.array([forward + turn, forward - turn]) + rng.normal(0, noise, 2)
    return np.clip(a, -1, 1)


def mixed_action(task, rng, p_hazard=0.5):
    """Per-call blend: seek goal or a hazard. Boosts unsafe coverage on default geometry."""
    return hazard_seek_action(task, rng) if rng.random() < p_hazard else goal_seek_action(task, rng)


def point_steer(task, want, thrust=1.0):
    """Point robot (action=[thrust along heading, turn rate]): turn toward `want` (world dir) and thrust
    along it, using reverse thrust when the target is behind (thrust acts along +-heading)."""
    m = np.asarray(task.agent.mat).reshape(3, 3); th = np.arctan2(m[1, 0], m[0, 0])
    e_f = _wrap(np.arctan2(want[1], want[0]) - th); e_b = _wrap(e_f + np.pi)
    e, s = (e_f, 1.0) if abs(e_f) <= abs(e_b) else (e_b, -1.0)
    return np.array([s * thrust if abs(e) < 0.6 else 0.0, np.clip(3 * e, -1, 1)])


def point_goal_action(task, rng, noise=0.3):
    """Full-thrust goal seek, no braking: momentum makes it overshoot and drift through hazards."""
    a = point_steer(task, np.asarray(task.goal.pos[:2]) - task.agent.pos[:2])
    return np.clip(a + rng.normal(0, noise, 2), -1, 1)


def point_dash_action(task, rng, brake_at, noise=0.2):
    """Dash at the nearest hazard at full thrust; once within `brake_at` of its edge, brake (thrust against
    velocity). Small brake_at -> hits at speed; large -> near-miss stops. Gives both sides of the
    velocity-dependent safe-set boundary."""
    p = task.agent.pos[:2]; hs = np.asarray([h[:2] for h in task.hazards.pos])
    j = np.linalg.norm(hs - p, axis=1).argmin(); edge = np.linalg.norm(hs[j] - p) - float(task.hazards.size)
    v = np.asarray(task.agent.vel[:2])
    a = point_steer(task, -v) if (edge < brake_at and np.linalg.norm(v) > 0.05) else point_steer(task, hs[j] - p)
    return np.clip(a + rng.normal(0, noise, 2), -1, 1)


def loiter_action(task, rng, forward=0.5, kturn=2.5, noise=0.5):
    """Approach the nearest hazard and linger in/around it: while outside, drive toward it;
    once inside (agent center within hazard), small circling actions to accumulate unsafe
    frames (approach->inside->exit dynamics, like dubins passing through an obstacle)."""
    x, y = float(task.agent.pos[0]), float(task.agent.pos[1])
    m = np.asarray(task.agent.mat).reshape(3, 3); th = np.arctan2(m[1, 0], m[0, 0])
    hs = np.asarray([h[:2] for h in task.hazards.pos]); d = np.linalg.norm(hs - [x, y], axis=1)
    j = d.argmin(); dist = d[j] - float(task.hazards.size)
    if dist < 0.02:  # inside/at the boundary -> tight circling to linger
        a = np.array([0.4, -0.1]) + rng.normal(0, noise, 2)
    else:
        e = _wrap(np.arctan2(hs[j][1] - y, hs[j][0] - x) - th)
        a = np.array([forward + np.clip(kturn * e, -1, 1), forward - np.clip(kturn * e, -1, 1)]) + rng.normal(0, noise * 0.5, 2)
    return np.clip(a, -1, 1)
