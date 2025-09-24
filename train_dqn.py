# ================================
# File: train_dqn.py (GPU ready)
# ================================
from __future__ import annotations
import os
import pickle
import random
import math
from collections import deque
from typing import Tuple, Dict
from contextlib import nullcontext

from cards import id_to_card
from game4p import GameState

from utils.HeuristicAgent import HeuristicAgent

import torch

print("CUDA disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome GPU:", torch.cuda.get_device_name(0))
    print("Memoria allocata:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("Memoria riservata:", torch.cuda.memory_reserved(0) / 1024**2, "MB")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

from game4p import deal, step, GameState, TEAM_OF_SEAT
from rules import score_cards_thirds
from obs.encoder import encode_state, update_void_flags
from cards import id_to_card

# ================================
# Config
# ================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Parametri consigliati per L40
EPISODES = 20000
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 512
REPLAY_CAP = 1_000_000
TARGET_SYNC = 5000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 20_000
PRINT_EVERY = 500
CHECKPOINT_EVERY = 5000
TRICK_SHAPING_SCALE = 1.0 / 8.0

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

"""
# ================================
# Euristica
# ================================
def euristica(state: GameState, legal_idx: list[int]) -> int:
    carte_uscite = set()
    for _, cid in state.trick.plays:
        carte_uscite.add(cid)
    for team_cards in state.captures_team.values():
        carte_uscite.update(team_cards)

    if not state.trick.plays:
        return min(legal_idx, key=lambda c: id_to_card(c).strength)

    lead_suit = id_to_card(state.trick.plays[0][1]).suit
    same_suit = [cid for cid in legal_idx if id_to_card(cid).suit == lead_suit]

    if same_suit:
        for cid in same_suit:
            card = id_to_card(cid)
            if card.rank == "A":
                due = any(id_to_card(c).rank == "2" and id_to_card(c).suit == lead_suit for c in carte_uscite)
                tre = any(id_to_card(c).rank == "3" and id_to_card(c).suit == lead_suit for c in carte_uscite)
                if not (due and tre):
                    other = [c for c in same_suit if id_to_card(c).rank != "A"]
                    if other:
                        return min(other, key=lambda c: id_to_card(c).strength)
        return min(same_suit, key=lambda c: id_to_card(c).strength)
    else:
        min_strength = min(id_to_card(c).strength for c in legal_idx)
        candidate = [c for c in legal_idx if id_to_card(c).strength == min_strength]

        tutte_le_carte = set(range(40))
        carte_in_giro = tutte_le_carte - carte_uscite
        min_strength_in_circolo = min(id_to_card(c).strength for c in carte_in_giro)
        if min_strength_in_circolo < min_strength:
            alternative = [c for c in legal_idx if id_to_card(c).strength > min_strength]
            if alternative:
                candidate = alternative

        if len(candidate) > 1:
            seme_counts = {}
            for _, cid in state.trick.plays:
                seme_counts[id_to_card(cid).suit] = seme_counts.get(id_to_card(cid).suit, 0) + 1
            for team_cards in state.captures_team.values():
                for cid in team_cards:
                    seme_counts[id_to_card(cid).suit] = seme_counts.get(id_to_card(cid).suit, 0) + 1
            return max(candidate, key=lambda c: seme_counts.get(id_to_card(c).suit, 0))
        else:
            return candidate[0]
"""

# ================================
# Training
# ================================
def train(resume_from: str | None = None):
    in_dim = feature_dim()
    policy = DQNNet(in_dim).to(DEVICE)
    target = DQNNet(in_dim).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()

    reward_history = []
    opt = optim.Adam(policy.parameters(), lr=LR)
    rb = Replay(REPLAY_CAP)
    opt_steps = 0
    start_ep = 1

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        policy.load_state_dict(checkpoint["model"])
        target.load_state_dict(checkpoint["target"])
        opt_steps = checkpoint.get("opt_steps", 0)
        start_ep = checkpoint.get("episode", 0) + 1
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
            seat = state.current_player
            x, mask = encode_state(state, seat, void_flags)
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            eps = epsilon(opt_steps)

            legal_idx = torch.nonzero(mask[0]).view(-1).tolist()

            if ep < 500:
                if random.random() < 0.7:
                    action = HeuristicAgent.choose_action(state, legal_idx)
                else:  # 30% casuale
                    action = random.choice(legal_idx)
            elif ep < 3000:  # Fase 2: euristica pura
                    action = HeuristicAgent.choose_action(state, legal_idx)

                    assert action in legal_idx, f"Azione {action} non valida! Legal idx: {legal_idx}"

            else:
                if random.random() < eps:
                    action = random.choice(legal_idx)
                else:
                    with torch.no_grad():
                        q = policy(x, mask)
                        action = int(q.argmax(dim=1).item())

            update_void_flags(void_flags, state, seat, action)
            prev_captures = {0:list(state.captures_team[0]),1:list(state.captures_team[1])}
            next_state, rewards, done, info = step(state, action)

            # Logga il punteggio della mano (differenza tra team)
            my_team_score = rewards[0] - rewards[1]
            reward_log.append(my_team_score)

            # Reward shaping: terzi catturati
            trick_closed = next_state.tricks_played > state.tricks_played
            r_shape = 0.0
            if trick_closed:
                new0 = score_cards_thirds(next_state.captures_team[0]) - score_cards_thirds(prev_captures[0])
                new1 = score_cards_thirds(next_state.captures_team[1]) - score_cards_thirds(prev_captures[1])
                # if new0>0 or new1>0: AGGIUNGE SOLO I TRICK CHE DANNO PUNTI
                r_team_shape = (new0 - new1) * TRICK_SHAPING_SCALE
                my_team = TEAM_OF_SEAT[seat]
                r_shape = r_team_shape if my_team == 0 else -r_team_shape

                # Aumenta peso ultima mano
                current_hand = next_state.tricks_played
                if current_hand == 10:
                    r_shape *= 2.0

                total_tricks += 1
                # Salva reward intermedio per logging
                reward_log.append(r_shape)

            # Reward finale: differenza di punteggio tra i team (solo a fine mano)
            if done:
                my_team = TEAM_OF_SEAT[seat]
                other_team = 1 - my_team
                r_final = float(rewards[my_team]) - float(rewards[other_team])
                reward_history.append(r_final)  # log reward finale
                if len(reward_history) > 1000:  # tieni solo ultimi 1000 episodi
                    reward_history.pop(0)
            else:
                r_final = 0.0

            # Reward totale = shaping intermedio + finale
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
            avg_trick_reward = sum(reward_log) / len(reward_log) if reward_log else 0.0
            print(f"[ep {ep}] replay={len(rb)} opt_steps={opt_steps} eps={epsilon(opt_steps):.3f} "
                  f"hands={total_hands} tricks={total_tricks} avg_final_reward={avg_final_reward:.2f} "
                  f"avg_trick_reward={avg_trick_reward:.2f}")

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
            with open(rb_file, "wb") as f:
                pickle.dump(rb.buf, f)
            print(f"[ep {ep}] checkpoint salvato: modello + replay buffer")

    torch.save({
        "model": policy.state_dict(),
        "config": {"in_dim": in_dim, "hidden": 256}
    }, "dqn_tressette_shared.pt")
    print("Salvato modello finale: dqn_tressette_shared.pt")

# ================================
# Entrypoint
# ================================
if __name__ == "__main__":
    train(resume_from=None)
