# ================================
# File: watch_game.py
# ================================
import random
import torch
import time

from game4p import deal, step, TEAM_OF_SEAT
from obs.encoder import encode_state, update_void_flags
from cards import id_to_card
from utils.HeuristicAgent import HeuristicAgent
from train_dqn import DQNNet, feature_dim, DEVICE

# ================================
# Setup
# ================================
PLAYERS = ["dqn", "random", "dqn", "random"]
CKPT = "dqn_tressette_shared.pt"

policy = None
if "dqn" in PLAYERS:
    policy = DQNNet(feature_dim()).to(DEVICE)
    checkpoint = torch.load(CKPT, map_location=DEVICE)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

# ================================
# Funzioni
# ================================
def choose_action(seat, state, void_flags):
    player_type = PLAYERS[seat]
    x, mask = encode_state(state, seat, void_flags)
    x, mask = x.to(DEVICE), mask.to(DEVICE)
    legal_idx = torch.nonzero(mask[0]).view(-1).tolist()

    if player_type == "dqn":
        with torch.no_grad():
            q = policy(x, mask)
            action = int(q.argmax(dim=1).item())
    elif player_type == "heuristic":
        action = HeuristicAgent.choose_action(state, legal_idx)
    else:
        action = random.choice(legal_idx)
    return action


def play_one_game(verbose=True, seed=None):
    if seed is None:
        seed = int(time.time() * 1e6) % (2**32 - 1)
    rng = random.Random(seed)

    state = deal(leader=rng.randint(0, 3), rng=rng)
    void_flags = [[0]*4 for _ in range(4)]
    done = False

    if verbose:
        print(f"[DEBUG] Seed partita: {seed}")

    while not done:
        seat = state.current_player
        action = choose_action(seat, state, void_flags)
        if verbose:
            print(f"Giocatore {seat} ({PLAYERS[seat]}) gioca {id_to_card(action)}")

        update_void_flags(void_flags, state, seat, action)
        state, rewards, done, _ = step(state, action)

    if verbose:
        print("=== Fine partita ===")
        print(f"Team 0 (seat 0+2) → {rewards[0]} punti")
        print(f"Team 1 (seat 1+3) → {rewards[1]} punti")
        if rewards[0] > rewards[1]:
            print("Vince TEAM 0")
        elif rewards[1] > rewards[0]:
            print("Vince TEAM 1")
        else:
            print("Pareggio!")

    return rewards

# ================================
# Main
# ================================
if __name__ == "__main__":
    print("=== DEMO PARTITA SINGOLA ===")
    play_one_game(verbose=True)

    N_MATCHES = 1000
    wins_team0 = wins_team1 = draws = 0

    print(f"\n=== TORNEO SU {N_MATCHES} PARTITE ===")

    for i in range(N_MATCHES):
        rewards = play_one_game(verbose=False)

        # Arrotonda i punteggi all’intero minore
        t0 = int(rewards[0])
        t1 = int(rewards[1])

        if t0 == t1:
            draws += 1
        elif t0 > t1:
            wins_team0 += 1
        else:
            wins_team1 += 1

    print(f"Team0 (seat 0+2: {PLAYERS[0]} + {PLAYERS[2]}) → vittorie: {wins_team0}/{N_MATCHES}")
    print(f"Team1 (seat 1+3: {PLAYERS[1]} + {PLAYERS[3]}) → vittorie: {wins_team1}/{N_MATCHES}")
    print(f"Pareggi: {draws}/{N_MATCHES}")
