# ================================
# File: train_dqn.py
# ================================
from __future__ import annotations
import random
import math
from collections import deque
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cards import id_to_card
from rules import score_cards_thirds
from game4p import deal, step, legal_action_mask, GameState, TEAM_OF_SEAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Configurazione
# ================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

EPISODES = 50_000          # aumenta in vero training (milioni di mani)
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 512
REPLAY_CAP = 200_000
TARGET_SYNC = 2_000        # ogni N ottimizzazioni
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000  # decadimento epsilon
PRINT_EVERY = 500

# Reward shaping (al completamento di una presa)
# valore per terzi catturati nella presa (es. 3 terzi = +1.0 shaping)
TRICK_SHAPING_SCALE = 1.0 / 3.0

# ================================
# Rete DQN condivisa
# ================================
class DQNNet(nn.Module):
    """Rete DQN: input tabellare -> Q(40)."""
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
        # Salviamo tensori già sul device per velocità
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
# Feature engineering (inline)
# ================================
def update_void_flags_before_step(void_flags, state: GameState, acting_seat: int, card_id: int):
    """Aggiorna i void flags in base alla giocata corrente:
       se il trick ha già una carta e io non seguo il seme d'uscita -> sono void su quel seme.
       NB: usiamo lo stato PRIMA dello step per leggere il seme di uscita corretto.
    """
    if len(state.trick.plays) == 0:
        return
    lead_suit = id_to_card(state.trick.plays[0][1]).suit
    played_suit = id_to_card(card_id).suit
    if played_suit != lead_suit:
        void_flags[acting_seat][lead_suit] = 1

def encode_state(state: GameState, seat: int, void_flags) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crea feature + action mask per il seat.
    Feature:
      - mano (40)
      - carte uscite (40): prese + carte sul tavolo
      - void flags (4x4)
      - seat_id (4), ally_pos (4)
      - segnali (4 giocatori x 4 semi x 3 segnali) -> flatten
    """
    # mano
    hand = torch.zeros(40)
    for c in state.hands[seat]:
        hand[c] = 1.0

    # carte uscite (prese + trick corrente)
    played = torch.zeros(40)
    for _, cid in state.trick.plays:
        played[cid] = 1.0
    for team_cards in state.captures_team.values():
        for c in team_cards:
            played[c] = 1.0

    # void flags
    voids = torch.tensor(void_flags, dtype=torch.float32).flatten()

    # seat_id e compagno
    seat_id = torch.zeros(4); seat_id[seat] = 1.0
    ally_pos = torch.zeros(4); ally_pos[(seat + 2) % 4] = 1.0

    # segnali one-hot (4x4x3)
    signals_tensor = torch.zeros(4, 4, 3)
    for s, sig in state.signals.items():
        seme = sig["suit"]
        if sig["signal"] == "volo":
            signals_tensor[s, seme, 0] = 1
        elif sig["signal"] == "striscio":
            signals_tensor[s, seme, 1] = 1
        elif sig["signal"] == "busso":
            signals_tensor[s, seme, 2] = 1

    x = torch.cat([hand, played, voids, seat_id, ally_pos, signals_tensor.flatten()])
    x = x.unsqueeze(0).to(DEVICE)

    # mask
    mask = torch.tensor(legal_action_mask(state), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return x, mask

def feature_dim() -> int:
    # mano(40) + uscite(40) + voids(16) + seat(4) + ally(4) + segnali(4*4*3=48)
    return 40 + 40 + 16 + 4 + 4 + 48

# ================================
# Epsilon schedule
# ================================
def epsilon(step: int) -> float:
    # esponenziale liscia
    ratio = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_END + (EPS_START - EPS_END) * math.exp(-3.0 * ratio)

# ================================
# Training loop (self-play 4 seats)
# ================================
def train():
    in_dim = feature_dim()
    policy = DQNNet(in_dim, hidden=256, n_actions=40).to(DEVICE)
    target = DQNNet(in_dim, hidden=256, n_actions=40).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = optim.Adam(policy.parameters(), lr=LR)
    rb = Replay(REPLAY_CAP)

    opt_steps = 0
    total_tricks = 0
    total_hands = 0

    for ep in range(1, EPISODES + 1):
        state = deal(rng=random.Random(SEED + ep), leader=ep % 4)
        # void flags 4x4 (giocatore x seme)
        void_flags = [[0, 0, 0, 0] for _ in range(4)]

        # per shaping: terzi catturati fino ad ora per team
        thirds_captured = {0: 0, 1: 0}

        done = False
        last_team_thirds = {0: 0, 1: 0}  # tracking delta per presa

        while not done:
            seat = state.current_player
            x, mask = encode_state(state, seat, void_flags)

            # epsilon-greedy con maschera
            eps = epsilon(opt_steps)
            if random.random() < eps:
                legal_idx = torch.nonzero(mask[0]).view(-1).tolist()
                action = random.choice(legal_idx)
            else:
                with torch.no_grad():
                    q = policy(x, mask)
                    action = int(q.argmax(dim=1).item())

            # Prima dello step aggiorniamo i void flags se non seguiamo il seme
            update_void_flags_before_step(void_flags, state, seat, action)

            # Salva stato per calcolare shaping delta
            prev_captures = {0: list(state.captures_team[0]), 1: list(state.captures_team[1])}

            # Esegui la mossa
            next_state, rewards, done, info = step(state, action)

            # Shaping: se la presa si è chiusa in questo step, i terzi catturati aumentano
            # (possiamo dedurlo perché il trick è stato resettato: len(next_state.trick.plays)==0 e
            # in prev/next captures cambia)
            trick_closed = (len(next_state.trick.plays) == 0) and (state.trick.leader != next_state.trick.leader)
            r_shape = 0.0
            if trick_closed:
                # terzi nuovi per ciascun team
                new0 = score_cards_thirds(next_state.captures_team[0]) - score_cards_thirds(prev_captures[0])
                new1 = score_cards_thirds(next_state.captures_team[1]) - score_cards_thirds(prev_captures[1])
                # shaping alla squadra che ha catturato nella presa appena conclusa
                if new0 > 0 or new1 > 0:
                    r_team_shape = (new0 - new1) * TRICK_SHAPING_SCALE  # differenziale: positivo se ha preso team0
                    # reward "di squadra" per il seat corrente (che appartiene a team_of_seat)
                    my_team = TEAM_OF_SEAT[seat]
                    # se my_team==0, r_shape = +r_team_shape; se 1, r_shape = -r_team_shape
                    r_shape = r_team_shape if my_team == 0 else -r_team_shape
                    total_tricks += 1

            # Reward finale di mano (team reward) — consegnato solo a fine mano
            r_final = 0.0
            if done:
                my_team = TEAM_OF_SEAT[seat]
                r_final = float(rewards[my_team])

            # reward totale per la transizione
            r = r_shape + r_final

            # Next features (stessa prospettiva di seat — semplice ma funziona con IQL)
            x_next, mask_next = encode_state(next_state, seat, void_flags)

            # Push nel replay
            rb.push(x, action, r, x_next, float(done), mask_next)

            state = next_state

            # Ottimizzazione
            if len(rb) >= BATCH_SIZE:
                s, a, r_b, s2, d, m2 = rb.sample(BATCH_SIZE)
                q = policy(s) .gather(1, a)
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
            print(f"[ep {ep}] replay={len(rb)} opt_steps={opt_steps} eps={epsilon(opt_steps):.3f} "
                  f"hands={total_hands} tricks={total_tricks}")

    # Salvataggio finale
    torch.save({"model": policy.state_dict(),
                "config": {
                    "in_dim": in_dim,
                    "hidden": 256
                }},
               "dqn_tresette_shared.pt")
    print("Salvato modello in dqn_tresette_shared.pt")

# ================================
# Entrypoint
# ================================
if __name__ == "__main__":
    train()
