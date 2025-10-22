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

EPISODES = 700000 #Dovrebbero bastare per superare heuristics, servirebbe spingere a 2M
GAMMA = 0.99 #Vicino a 1 per una visione a lungo termine -> Non vincere la mano ma la partita
LR = 3e-4 #Precisione, abbastanza preciso ma non instabile
BATCH_SIZE = 512 #Esperienze estratte dal replay buffer ad ogni update
REPLAY_CAP = 1_000_000 #Capacità replay buffer
TARGET_SYNC = 2000 #Frequenza aggiornamento target network
EPS_START = 1.0 #Epsilon Iniziale -> mosse casuali
EPS_END = 0.05 #Epsilon Finale -> determinismo
EPS_DECAY_STEPS = 200_000 #Episodi per andare da inizio a fine
PRINT_EVERY = 15000
CHECKPOINT_EVERY = 15000

# ================================
# DQN
# ================================
"Creo rete DQN - Funzioni base - feature dim in base a numero carte ,trick ecc -forward = propagazione in avanti"
"Replay, per coda che elimina le memorie più datate"
class DQNNet(nn.Module):
    "Double layer + layer output"
    "Layer = Strato di neuroni, 1 Layer per estrazione + secondo combinazione feature"
    def __init__(self, input_dim: int, hidden: int = 256, n_actions: int = 40):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)


    "Forward = Cuore -> Capisce i pattern, stima Q-Values e può essere usato per decisioni o addestramento dato che genera i Q Values"
    def forward(self, x:     torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
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

    # Ogni elemento salvato nel replay buffer è una tupla con 7 campi:
    # (
    #   s,        # stato corrente (torch.Tensor): rappresentazione numerica dell'ambiente osservato dal giocatore
    #   m,        # maschera azioni legali (torch.Tensor): 1 dove la carta è giocabile, 0 dove non lo è
    #   a,        # azione eseguita (int): indice della carta giocata dal giocatore
    #   r,        # reward immediato (float): punteggio ottenuto dopo aver eseguito l’azione
    #   s_next,   # stato successivo (torch.Tensor): nuova osservazione dell’ambiente dopo la mossa
    #   m_next,   # maschera azioni legali nello stato successivo (torch.Tensor)
    #   done      # flag di terminazione (float)
    # )
    #
    # Queste tuple rappresentano esperienze complete di tipo (s, a, r, s_next, done),
    # con l'aggiunta delle maschere m e m_next per indicare le azioni consentite.
    # Il replay buffer funge da memoria circolare per riutilizzare esperienze passate
    # durante l’addestramento e ridurre la correlazione temporale tra i dati.

    def sample(self, B):  # Estrae un mini-batch casuale di B esperienze dal replay buffer
        batch = random.sample(self.buf, B)  # Seleziona B tuple casuali dal buffer (decorrelazione temporale)
        s, m, a, r, s2, m2, d = zip(
            *batch)  # Divide il batch in 7 liste: stati, maschere, azioni, reward, next state, next mask, done
        s = torch.cat(s, dim=0).to(DEVICE)  # Unisce tutti gli stati in un tensore [B, N] e li sposta su GPU/CPU
        m = torch.cat(m, dim=0).to(DEVICE)  # Unisce tutte le maschere di azione e le sposta su GPU/CPU
        a = torch.tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1)  # Converte le azioni in tensore intero [B,1]
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # Converte i reward in tensore float [B,1]
        s2 = torch.cat(s2, dim=0).to(DEVICE)  # Unisce tutti gli stati successivi e li porta su GPU/CPU
        m2 = torch.cat(m2, dim=0).to(DEVICE)  # Unisce tutte le maschere successive e le porta su GPU/CPU
        d = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(
            1)  # Converte i flag done (episodio finito = 1) in tensore float [B,1]
        return s, m, a, r, s2, m2, d  # Restituisce tutti i tensori del batch pronti per l’aggiornamento DQN

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
    "Creo replay buffer"
    rb = Replay(REPLAY_CAP)
    opt_steps = 0
    start_ep = 1
    "Memorizzo i reward finali"
    reward_history = []
    total_tricks = 0
    total_hands = 0

    "Resume da checkpoint"
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        policy.load_state_dict(checkpoint["model"])
        "Nel caso esistono, carichiamo i pesi"
        if "target" in checkpoint:
            target.load_state_dict(checkpoint["target"])
        else:
            target.load_state_dict(checkpoint["model"])  # fallback

        opt_steps = checkpoint.get("opt_steps", 0)
        start_ep = checkpoint.get("episode", 0) + 1
        rb_path = resume_from.replace(".pt", ".pkl")
        if os.path.exists(rb_path):
            rb.buf = deque(torch.load(rb_path), maxlen=REPLAY_CAP)
        print(f"Ripreso training da episodio {start_ep}, opt_steps={opt_steps}, replay={len(rb)}")

    for ep in range(start_ep, EPISODES + 1):
        "Applico funzione deal per assegnare le carte"
        state = deal(leader=ep % 4)
        "Flag per i semi che i giocatori non hanno"
        void_flags = [[0]*4 for _ in range(4)]
        reward_log = []
        done = False

        while not done:
            "Recuperiamo player corrente"
            seat = state.current_player
            "Da stato a tensore Pytorch"
            x, mask = encode_state(state, seat, void_flags)
            x, mask = x.to(DEVICE), mask.to(DEVICE)

            "Epsilon attuale , da 1 a valore baasso, es potremmo mettere 0.05, quando si abbassa non sto più esplorando ma gioco in maniera deterministica"
            eps = epsilon(opt_steps)
            "Solita maschera azione legali"
            legal_idx = torch.nonzero(mask[0]).view(-1).tolist()
            action = legal_idx[0] if legal_idx else 0

            "Scelgo politica training"
            if ep < 1000:
                if random.random() < 0.7:
                    action = HeuristicAgent.choose_action(state, legal_idx)
                else:
                    action = random.choice(legal_idx)
            elif ep < 600000:
                action = HeuristicAgent.choose_action(state, legal_idx)
            else:
                if random.random() < eps:
                    action = random.choice(legal_idx)
                else:
                    "Nel caso scelgo Q-Value più alto"
                    with torch.no_grad():
                        q = policy(x, mask).squeeze(0)
                        q_legal = q[legal_idx]
                        best_idx = torch.argmax(q_legal).item()
                        action = legal_idx[best_idx]

            "Fallback"
            if action not in legal_idx:
                print(f"[WARNING] Azione {action} non valida per giocatore {seat}. Legal idx: {legal_idx}")
                action = random.choice(legal_idx)

            update_void_flags(void_flags, state, seat, action)
            prev_captures = {0: list(state.captures_team[0]), 1: list(state.captures_team[1])}
            next_state, rewards, done, _ = step(state, action)

            "Calcolo reward"
            r_shape = compute_reward(state, next_state, seat, action, prev_captures)

            "Calcolo punti finale "
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

            "Creo transazioni e le aggiungo al replay buffer"
            x_next, mask_next = encode_state(next_state, seat, void_flags)
            x_next, mask_next = x_next.to(DEVICE), mask_next.to(DEVICE)
            rb.push(x, mask, action, r, x_next, mask_next, float(done))
            state = next_state

            "Calcolo Q"
            if len(rb) >= BATCH_SIZE:
                s, m, a, r_b, s2, m2, d = rb.sample(BATCH_SIZE)
                with amp_autocast():
                    q = policy(s, m).gather(1, a)
                    with torch.no_grad():
                        q2 = target(s2, m2).max(dim=1, keepdim=True)[0]
                        y = r_b + (1.0 - d) * GAMMA * q2
                    loss = F.mse_loss(q, y)

                "Aggiornamento parametri"
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

        "Print finali"
        if ep % PRINT_EVERY == 0:
            avg_final_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0
            print(f"[ep {ep}] replay={len(rb)} opt_steps={opt_steps} eps={eps:.3f} "
                  f"hands={total_hands} avg_final_reward={avg_final_reward:.2f}")

            # Salvataggio checkpoint periodico
            if ep % CHECKPOINT_EVERY == 0:
                ckpt_file = f"dqn_tressette_checkpoint_ep{ep}.pt"
                rb_file = f"dqn_tressette_checkpoint_ep{ep}.pkl"

                # Rimuovi i vecchi checkpoint per non saturare lo spazio
                for f in os.listdir("."):
                    if (
                            f.startswith("dqn_tressette_checkpoint_ep")
                            and (f.endswith(".pt") or f.endswith(".pkl"))
                            and f not in [ckpt_file, rb_file]
                    ):
                        os.remove(f)

                # Salva nuovo checkpoint
                torch.save({
                    "model": policy.state_dict(),
                    "target": target.state_dict(),
                    "opt_steps": opt_steps,
                    "episode": ep,
                    "config": {"in_dim": in_dim, "hidden": 256}
                }, ckpt_file)

                torch.save(list(rb.buf), rb_file)
                print(f"[ep {ep}] ✅ Checkpoint aggiornato: {ckpt_file}")

    # Fine training → salva modello finale
    final_model = f"dqn_tressette_ep{EPISODES}.pt"
    torch.save({
        "model": policy.state_dict(),
        "config": {"in_dim": in_dim, "hidden": 256}
    }, final_model)
    print(f"🏁 Training completato. Modello finale salvato: {final_model}")

# ================================
# Entrypoint con resume da checkpoint
# ================================
if __name__ == "__main__":
    CHECKPOINT = "dqn_tressette_checkpoint_ep390000.pt"
    train(resume_from=CHECKPOINT)


