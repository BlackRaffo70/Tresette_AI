# ================================
# File: train_dqn.py
# ================================
from __future__ import annotations
import random
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rules import score_cards_thirds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# ================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 512
REPLAY_CAP = 200_000
EPS_START = 1.0
EPS_END = 0.05
PRINT_EVERY = 500

TRICK_SHAPING_SCALE = 1.0 / 3.0

# ================================
# ================================
class DQNNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, n_actions: int = 40):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q = self.out(h)
        if mask is not None:
            q = q.masked_fill(mask == 0, -1e9)
        return q

# ================================
# Replay Buffer
# ================================
class Replay:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s_next, done, mask_next):
        self.buf.append((s.detach(), a, r, s_next.detach(), done, mask_next.detach()))

    def sample(self, B):
        batch = random.sample(self.buf, B)
        s, a, r, s2, d, m2 = zip(*batch)
        s = torch.cat(s, dim=0)
        a = torch.tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        s2 = torch.cat(s2, dim=0)
        d = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        m2 = torch.cat(m2, dim=0)
        return s, a, r, s2, d, m2

    def __len__(self):
        return len(self.buf)

# ================================
# ================================
def feature_dim() -> int:
    return 40 + 40 + 16 + 4 + 4 + 48

def epsilon(step: int) -> float:
    ratio = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_END + (EPS_START - EPS_END) * math.exp(-3.0 * ratio)

# ================================
# ================================
    in_dim = feature_dim()
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = optim.Adam(policy.parameters(), lr=LR)
    rb = Replay(REPLAY_CAP)
    opt_steps = 0
    total_tricks = 0
    total_hands = 0

        state = deal(rng=random.Random(SEED + ep), leader=ep % 4)
        done = False

        while not done:
            seat = state.current_player
            x, mask = encode_state(state, seat, void_flags)
            eps = epsilon(opt_steps)
            if random.random() < eps:
                legal_idx = torch.nonzero(mask[0]).view(-1).tolist()
                action = random.choice(legal_idx)
            else:
                with torch.no_grad():
                    q = policy(x, mask)
                    action = int(q.argmax(dim=1).item())


            prev_captures = {0:list(state.captures_team[0]),1:list(state.captures_team[1])}
            next_state, rewards, done, info = step(state, action)

            trick_closed = (len(next_state.trick.plays) == 0) and (state.trick.leader != next_state.trick.leader)
            r_shape = 0.0
            if trick_closed:
                new0 = score_cards_thirds(next_state.captures_team[0]) - score_cards_thirds(prev_captures[0])
                new1 = score_cards_thirds(next_state.captures_team[1]) - score_cards_thirds(prev_captures[1])
                if new0>0 or new1>0:
                    my_team = TEAM_OF_SEAT[seat]
                    r_shape = r_team_shape if my_team == 0 else -r_team_shape
                    total_tricks += 1

            r = r_shape + r_final

            x_next, mask_next = encode_state(next_state, seat, void_flags)
            rb.push(x, action, r, x_next, float(done), mask_next)
            state = next_state

            if len(rb) >= BATCH_SIZE:
                s,a,r_b,s2,d,m2 = rb.sample(BATCH_SIZE)
                q = policy(s).gather(1, a)
                with torch.no_grad():
                    q2 = target(s2, m2).max(dim=1, keepdim=True)[0]
                    y = r_b + (1.0 - d) * GAMMA * q2
                loss = F.mse_loss(q, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt_steps += 1
                if opt_steps % TARGET_SYNC == 0:
                    target.load_state_dict(policy.state_dict())

        total_hands += 1

        if ep % PRINT_EVERY == 0:

    # Salvataggio finale

# ================================
# Entrypoint
# ================================
if __name__ == "__main__":
