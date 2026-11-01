"""Self-check: lexicographic soft cost ranks safe < grazing < deep, even when the safe path is farther from the goal."""
import sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
import torch
import safe_solver as s


class FakeJEPA:
    def parameters(self):
        return iter([torch.zeros(1)])

    def get_cost(self, info, a):
        info["predicted_emb"] = torch.zeros(1, 3, 1 + 4, 1)  # H=1 history + 4 imagined
        return torch.tensor([[5., 1., 1.]])                 # safe path is the worst on goal


head = lambda z: torch.tensor([[[.1] * 4, [.1, -.01, .1, .1], [-.5] * 4]])  # safe, grazing, deep
c = s._SoftCostJEPA(FakeJEPA(), head, 1000., 0., 1e4).get_cost(
    {"pixels": torch.zeros(1, 3, 1, 3, 2, 2)}, None)
print(c)
assert c[0, 0] < c[0, 1] < c[0, 2] and c[0, 1] > 1e4


# goal_cost="mean": a path that reaches the goal early and stays must beat one that only arrives at the last step
class FakeJEPA2(FakeJEPA):
    def get_cost(self, info, a):
        early = torch.tensor([3., 0., 0., 0.]); late = torch.tensor([3., 2., 1., 0.])   # 1-D latent, goal at 0
        info["predicted_emb"] = torch.cat([torch.zeros(1, 2, 1, 1), torch.stack([early, late]).view(1, 2, 4, 1)], 2)
        info["goal_emb"] = torch.zeros(1, 1, 1)                                          # per-env goal (B, T, D)
        return torch.tensor([[0., 0.]])                                                  # final-step cost: a tie
safe = lambda z: torch.ones(z.shape[:-1])
c = s._SoftCostJEPA(FakeJEPA2(), safe, 1000., 0., 1e4, goal_cost="mean").get_cost({"pixels": torch.zeros(1, 2, 1, 3, 2, 2)}, None)
print(c)
assert torch.allclose(c, torch.tensor([[9 / 4, 14 / 4]])) and c[0, 0] < c[0, 1]
print("ok")
