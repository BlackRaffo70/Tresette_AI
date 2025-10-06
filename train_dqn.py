# ================================
# File: train_dqn.py (GPU ready + Dense Reward + Resume 200k → 400k)
# ================================
from __future__ import annotations
import os
import random
import math
from collections import deque
from typing import Dict
from contextlib import nullcontext

from cards import id_to_card
from game4p import deal, step, TEAM_OF_SEAT
from rules import score_cards_thirds
from obs.encoder import encode_state, update_void_flags
from utils.HeuristicAgent import HeuristicAgent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("CUDA disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome GPU:", torch.cuda.get_device_name(0))
    print("Memoria allocata:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("Memoria riservata:", torch.cuda.memory_reserved(0) / 1024**2, "MB")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Device selection (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    AMP_ENABLED = True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    AMP_ENABLED = False
else:
    DEVICE = torch.device("cpu")
    AMP_ENABLED = False

# GradScaler / autocast compatibili
if AMP_ENABLED:
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    def amp_autocast():
        return autocast(dtype=torch.float16)
else:
    class _NoOpScaler:
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
        def unscale_(self, opt): pass
    scaler = _NoOpScaler()
    def amp_autocast():
        return nullcontext()

# ================================
# Config
# ================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

EPISODES = 500000
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 512
REPLAY_CAP = 5_000_000
TARGET_SYNC = 2000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000
PRINT_EVERY = 25000
CHECKPOINT_EVERY = 25000

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
            q = q.masked_fill(mask == 0, -1e4)
        return q

# ================================
# Replay Buffer
# ================================
class Replay:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)

    def push(self, s, m, a, r, s_next, m_next, done):
        self.buf.append((s.detach(), m.detach(), a, r, s_next.detach(), m_next.detach(), done))

    def sample(self, B):
        batch = random.sample(self.buf, B)
        s, m, a, r, s2, m2, d = zip(*batch)
        s  = torch.cat(s, dim=0).to(DEVICE)
        m  = torch.cat(m, dim=0).to(DEVICE)
        a  = torch.tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1)
        r  = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        s2 = torch.cat(s2, dim=0).to(DEVICE)
        m2 = torch.cat(m2, dim=0).to(DEVICE)
        d  = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        return s, m, a, r, s2, m2, d

    def __len__(self):
        return len(self.buf)

# ================================
# Feature dimension & epsilon
# ================================
def feature_dim() -> int:
    return 40 + 40 + 16 + 4 + 4 + 48

def epsilon(step: int) -> float:
    ratio = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_END + (EPS_START - EPS_END) * math.exp(-3.0 * ratio)

# ================================
# Dense Reward Function
# ================================
def compute_reward(state, next_state, seat, action, prev_captures):
    r = 0.0
    card = id_to_card(action)

    # penalizza carte forti giocate presto
    if card.rank == "A":
        r -= 0.2
    elif card.rank in ["2", "3"]:
        r -= 0.1
    else:
        r += 0.05

    # reward per chiusura trick
    trick_closed = next_state.tricks_played > state.tricks_played
    if trick_closed:
        my_team = TEAM_OF_SEAT[seat]

        # se ho preso il trick → +0.5, altrimenti -0.5
        if len(next_state.captures_team[my_team]) > len(state.captures_team[my_team]):
            r += 0.5
        else:
            r -= 0.5

        # bonus se ho catturato 2, 3 o Asso
        captured = set(next_state.captures_team[my_team]) - set(prev_captures[my_team])
        for cid in captured:
            c = id_to_card(cid)
            if c.rank in ["2", "3", "A"]:
                r += 0.3

    return r

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

    reward_history = []
    total_tricks = 0
    total_hands = 0

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        policy.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        opt_steps = checkpoint.get("opt_steps", 0)
        start_ep = checkpoint.get("episode", 0) + 1
        rb_path = resume_from.replace(".pt", ".pkl")
        if os.path.exists(rb_path):
            rb.buf = deque(torch.load(rb_path), maxlen=REPLAY_CAP)
        print(f"Ripreso training da episodio {start_ep}, opt_steps={opt_steps}, replay={len(rb)}")

    for ep in range(start_ep, EPISODES + 1):
        state = deal(leader=ep % 4)
        void_flags = [[0]*4 for _ in range(4)]
        reward_log = []
        done = False

        while not done:
            seat = state.current_player
            x, mask = encode_state(state, seat, void_flags)
            x, mask = x.to(DEVICE), mask.to(DEVICE)

            eps = epsilon(opt_steps)
            legal_idx = torch.nonzero(mask[0]).view(-1).tolist()
            action = legal_idx[0] if legal_idx else 0

            if ep < 1000:
                if random.random() < 0.7:
                    action = HeuristicAgent.choose_action(state, legal_idx)
                else:
                    action = random.choice(legal_idx)
            elif ep < 10000:
                action = HeuristicAgent.choose_action(state, legal_idx)
            else:
                if random.random() < eps:
                    action = random.choice(legal_idx)
                else:
                    with torch.no_grad():
                        q = policy(x, mask).squeeze(0)
                        q_legal = q[legal_idx]
                        best_idx = torch.argmax(q_legal).item()
                        action = legal_idx[best_idx]

            if action not in legal_idx:
                print(f"[WARNING] Azione {action} non valida per giocatore {seat}. Legal idx: {legal_idx}")
                action = random.choice(legal_idx)

            update_void_flags(void_flags, state, seat, action)
            prev_captures = {0: list(state.captures_team[0]), 1: list(state.captures_team[1])}
            next_state, rewards, done, _ = step(state, action)

            r_shape = compute_reward(state, next_state, seat, action, prev_captures)

            if done:
                my_team = TEAM_OF_SEAT[seat]
                other_team = 1 - my_team
                r_final = float(rewards[my_team]) - float(rewards[other_team])
                reward_history.append(r_final)
                if len(reward_history) > 1000:
                    reward_history.pop(0)
            else:
                r_final = 0.0

            r = r_shape + r_final

            x_next, mask_next = encode_state(next_state, seat, void_flags)
            x_next, mask_next = x_next.to(DEVICE), mask_next.to(DEVICE)
            rb.push(x, mask, action, r, x_next, mask_next, float(done))
            state = next_state

            if len(rb) >= BATCH_SIZE:
                s, m, a, r_b, s2, m2, d = rb.sample(BATCH_SIZE)
                with amp_autocast():
                    q = policy(s, m).gather(1, a)
                    with torch.no_grad():
                        q2 = target(s2, m2).max(dim=1, keepdim=True)[0]
                        y = r_b + (1.0 - d) * GAMMA * q2
                    loss = F.mse_loss(q, y)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt_steps += 1

                if opt_steps % TARGET_SYNC == 0:
                    target.load_state_dict(policy.state_dict())

        total_hands += 1

        if ep % PRINT_EVERY == 0:
            avg_final_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
            print(f"[ep {ep}] replay={len(rb)} opt_steps={opt_steps} eps={eps:.3f} "
                  f"hands={total_hands} avg_final_reward={avg_final_reward:.2f}")

        if ep % CHECKPOINT_EVERY == 0:
            ckpt_file = f"dqn_tressette_checkpoint_ep{ep}.pt"
            rb_file = f"dqn_tressette_checkpoint_ep{ep}.pkl"
            torch.save({
                "model": policy.state_dict(),
                "target": target.state_dict(),
                "opt_steps": opt_steps,
                "episode": ep,
                "config": {"in_dim": in_dim, "hidden": 256}
            }, ckpt_file)
            torch.save(list(rb.buf), rb_file)
            print(f"[ep {ep}] checkpoint salvato: {ckpt_file}")

    torch.save({
        "model": policy.state_dict(),
        "config": {"in_dim": in_dim, "hidden": 256}
    }, "dqn_tressette_shared.pt")
    print("Salvato modello finale: dqn_tressette_shared.pt")

# ================================
# Entrypoint con resume da checkpoint
# ================================
if __name__ == "__main__":
    CHECKPOINT = "dqn_tressette_checkpoint_ep250000.pt"
    train(resume_from=CHECKPOINT)