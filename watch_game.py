# ================================
# File: watch_game.py
# ================================
import random
import torch

from game4p import deal, step, TEAM_OF_SEAT
from obs.encoder import encode_state, update_void_flags
from cards import id_to_card
from utils.HeuristicAgent import HeuristicAgent
from train_dqn import DQNNet, feature_dim, DEVICE

# ================================
# Setup
# ================================
SEED = 123
random.seed(SEED)
torch.manual_seed(SEED)

# Carica un modello DQN già allenato (se vuoi vedere la rete giocare)
USE_DQN = False        # metti True se vuoi che il giocatore 0 usi la rete
CKPT = "dqn_tressette_checkpoint_ep5000.pt"

policy = None
if USE_DQN:
    policy = DQNNet(feature_dim()).to(DEVICE)
    checkpoint = torch.load(CKPT, map_location=DEVICE)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

# ================================
# Gioca una partita
# ================================
state = deal(leader=0)   # nuovo mazzo, giocatore 0 inizia
void_flags = [[0]*4 for _ in range(4)]
done = False

print("=== Inizio partita Tresette ===")

while not done:
    seat = state.current_player
    x, mask = encode_state(state, seat, void_flags)
    x, mask = x.to(DEVICE), mask.to(DEVICE)

    legal_idx = torch.nonzero(mask[0]).view(-1).tolist()

    if seat == 0:  # Giocatore 0 con DQN
        with torch.no_grad():
            q = policy(x, mask)
            action = int(q.argmax(dim=1).item())

    elif seat == 1:  # Giocatore 1 random
        action = random.choice(legal_idx)

    elif seat == 2:  # Giocatore 2 euristica
        action = HeuristicAgent.choose_action(state, legal_idx)

    else:  # Giocatore 3 euristica
        action = HeuristicAgent.choose_action(state, legal_idx)

    # Stampa la mossa
    card = id_to_card(action)
    print(f"Giocatore {seat} gioca {card}")

    # Aggiorna flags e stato
    update_void_flags(void_flags, state, seat, action)
    state, rewards, done, info = step(state, action)

print("=== Fine partita ===")
print("Punteggio finale:")
print(f"Team 0: {rewards[0]} punti")
print(f"Team 1: {rewards[1]} punti")

if rewards[0] > rewards[1]:
    print("Vince TEAM 0 (giocatori 0 e 2)")
elif rewards[1] > rewards[0]:
    print("Vince TEAM 1 (giocatori 1 e 3)")
else:
    print("Pareggio!")