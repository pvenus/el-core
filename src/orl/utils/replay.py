from collections import deque, namedtuple
import random, numpy as np, torch

Transition = namedtuple("Transition", ["s", "a", "r", "ns", "d"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append(Transition(s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = torch.from_numpy(np.stack([b.s for b in batch])).float()
        a = torch.tensor([b.a for b in batch]).long().unsqueeze(-1)
        r = torch.tensor([b.r for b in batch]).float().unsqueeze(-1)
        ns = torch.from_numpy(np.stack([b.ns for b in batch])).float()
        d = torch.tensor([b.d for b in batch]).float().unsqueeze(-1)
        return Transition(s, a, r, ns, d)

    def __len__(self): return len(self.buf)
