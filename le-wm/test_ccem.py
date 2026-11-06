"""Check the SafeDreamer CCEM elite rule in _SoftCostJEPA: with >= topk feasible candidates, the topk smallest scores are
the feasible candidates with the best goal cost; with fewer, they are the lowest-violation candidates (lower index on
ties). Run: python test_ccem.py"""
import torch
from safe_solver import _SoftCostJEPA


class FakeJEPA:
    def __init__(self, goal):
        self.goal = goal
    def get_cost(self, info, cand):
        return self.goal.clone()


head = lambda z: z[..., 0]                      # h = first latent coordinate; h < 0 means unsafe


def run(goal, h_steps, topk, fallback="count"):
    B, S, T = h_steps.shape                     # h of current state + T-1 imagined steps
    emb = torch.zeros(B, S, T, 2); emb[..., 0] = h_steps   # 1 context frame (H = 1): [:, :, H-1:] = current + imagined
    info = {"pixels": torch.zeros(B, S, 1), "predicted_emb": emb}
    m = _SoftCostJEPA(FakeJEPA(goal), head, 1000, 0.0, 1e4, ccem=True, ccem_fallback=fallback); m.topk = topk
    return torch.topk(m.get_cost(info, None), k=topk, dim=1, largest=False).indices[0].sort().values.tolist()


# 6 candidates, 3 steps (current + 2 imagined). Candidates 0, 2, 3, 5 feasible (no h < 0).
H = torch.tensor([[[1, 1, 1], [1, -1, 1], [1, 1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]]], dtype=torch.float)
goal = torch.tensor([[5.0, 0.0, 3.0, 1.0, 0.0, 2.0]])
assert run(goal, H, topk=2) == [3, 5], run(goal, H, topk=2)          # enough feasible: best goal among feasible (1.0, 2.0)
assert run(goal, H, topk=5) == [0, 1, 2, 3, 5], run(goal, H, topk=5)  # too few (4 < 5): lowest cost, then lower index
# "minh" fallback: feasible = h > 0 everywhere; too few feasible -> all feasible, then highest worst-case h.
H2 = torch.tensor([[[1, 1, 1], [1, -0.2, 1], [1, 1, 1], [1, -3, 1], [-1, -1, 1], [1, 1, 1]]], dtype=torch.float)
assert run(goal, H2, topk=2, fallback="minh") == [2, 5], run(goal, H2, topk=2, fallback="minh")  # enough: best J of {0,2,5}
assert run(goal, H2, topk=4, fallback="minh") == [0, 1, 2, 5], run(goal, H2, topk=4, fallback="minh")  # 3 feasible + best worst-case (1: -0.2)
assert run(goal, H2, topk=5, fallback="minh") == [0, 1, 2, 4, 5], run(goal, H2, topk=5, fallback="minh")  # then 4 (-1) before 3 (-3)
print("ccem rule ok")
