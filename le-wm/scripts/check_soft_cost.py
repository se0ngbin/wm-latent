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
print("ok")
