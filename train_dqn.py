# ================================
# File: train_dqn.py
# ================================
from __future__ import annotations
import os
import pickle
import random
import math
from collections import deque
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game4p import deal, step, GameState, TEAM_OF_SEAT
from rules import score_cards_thirds
from obs.encoder import encode_state, update_void_flags

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Config
# ================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

EPISODES = 50_000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 512
REPLAY_CAP = 200_000
TARGET_SYNC = 2_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000
PRINT_EVERY = 500
CHECKPOINT_EVERY = 10_000

TRICK_SHAPING_SCALE = 1.0 / 3.0

# ================================
# DQN
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
# Feature dimension & epsilon
# ================================
def feature_dim() -> int:
    # mano(40) + uscite(40) + voids(16) + seat(4) + ally(4) + segnali(48)
    return 40 + 40 + 16 + 4 + 4 + 48

def epsilon(step: int) -> float:
    ratio = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_END + (EPS_START - EPS_END) * math.exp(-3.0 * ratio)

# ================================
# Training
# ================================
def train(resume_from: str | None = None):
    in_dim = feature_dim()
    policy = DQNNet(in_dim).to(DEVICE)
    target = DQNNet(in_dim).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = optim.Adam(policy.parameters(), lr=LR)
    rb = Replay(REPLAY_CAP)
    opt_steps = 0
    start_ep = 1

    # Ripresa da checkpoint
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        policy.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        opt_steps = checkpoint.get("opt_steps", 0)
        start_ep = checkpoint.get("episode",0) + 1
        # replay buffer
        rb_path = resume_from.replace(".pt", ".pkl")
        if os.path.exists(rb_path):
            with open(rb_path, "rb") as f:
                rb.buf = pickle.load(f)
        print(f"Ripreso training da episodio {start_ep}, opt_steps={opt_steps}, replay={len(rb)}")

    total_tricks = 0
    total_hands = 0

    for ep in range(start_ep, EPISODES + 1):
        state = deal(rng=random.Random(SEED + ep), leader=ep % 4)
        void_flags = [[0]*4 for _ in range(4)]
        reward_log = []
        done = False

        while not done:
            # Logga il punteggio della mano (differenza tra team)
            my_team_score = rewards[0] - rewards[1]
            reward_log.append(my_team_score)

            seat = state.current_player
            x, mask = encode_state(state, seat, void_flags)
            eps = epsilon(opt_steps)

            # epsilon-greedy
            if random.random() < eps:
                legal_idx = torch.nonzero(mask[0]).view(-1).tolist()
                action = random.choice(legal_idx)
            else:
                with torch.no_grad():
                    q = policy(x, mask)
                    action = int(q.argmax(dim=1).item())

            # update void flags prima dello step
            update_void_flags(void_flags, state, seat, action)

            prev_captures = {0:list(state.captures_team[0]),1:list(state.captures_team[1])}
            next_state, rewards, done, info = step(state, action)

            # Reward shaping: terzi catturati
            trick_closed = (len(next_state.trick.plays) == 0) and (state.trick.leader != next_state.trick.leader)
            r_shape = 0.0
            if trick_closed:
                new0 = score_cards_thirds(next_state.captures_team[0]) - score_cards_thirds(prev_captures[0])
                new1 = score_cards_thirds(next_state.captures_team[1]) - score_cards_thirds(prev_captures[1])
                if new0>0 or new1>0:
                    r_team_shape = (new0-new1) * TRICK_SHAPING_SCALE
                    my_team = TEAM_OF_SEAT[seat]
                    r_shape = r_team_shape if my_team == 0 else -r_team_shape
                    total_tricks += 1

            # Reward finale: differenza di punteggio tra i team (solo a fine mano)
            if done:
                my_team = TEAM_OF_SEAT[seat]
                other_team = 1 - my_team
                r_final = float(rewards[my_team]) - float(rewards[other_team])
            else:
                r_final = 0.0

            # Reward totale = shaping intermedio + finale
            r = r_shape + r_final

            x_next, mask_next = encode_state(next_state, seat, void_flags)
            rb.push(x, action, r, x_next, float(done), mask_next)
            state = next_state

            # Ottimizzazione DQN
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
            avg_reward = sum(reward_log) / len(reward_log) if reward_log else 0
            print(f"[ep {ep}] replay={len(rb)} opt_steps={opt_steps} eps={epsilon(opt_steps):.3f} "
                  f"hands={total_hands} tricks={total_tricks} avg_score={avg_reward:.2f}")
            reward_log.clear()

            print(f"[ep {ep}] replay={len(rb)} opt_steps={opt_steps} eps={epsilon(opt_steps):.3f} hands={total_hands} tricks={total_tricks}")

        # Checkpoint automatico
        if ep % CHECKPOINT_EVERY == 0:
            ckpt_file = f"dqn_tresette_checkpoint_ep{ep}.pt"
            rb_file = f"dqn_tresette_checkpoint_ep{ep}.pkl"
            torch.save({
                "model": policy.state_dict(),
                "target": target.state_dict(),
                "opt_steps": opt_steps,
                "episode": ep,
                "config": {"in_dim": in_dim, "hidden": 256}
            }, ckpt_file)
            with open(rb_file, "wb") as f:
                pickle.dump(rb.buf, f)
            print(f"[ep {ep}] checkpoint salvato: modello + replay buffer")

    # Salvataggio finale
    torch.save({
        "model": policy.state_dict(),
        "config": {"in_dim": in_dim, "hidden": 256}
    }, "dqn_tresette_shared.pt")
    print("Salvato modello finale: dqn_tresette_shared.pt")

# ================================
# Entrypoint
# ================================
if __name__ == "__main__":
    train(resume_from=None)
