# ================================
# File: watch_game_parallel.py
# ================================
import random
import torch
import time
from tqdm import tqdm

from game4p import deal, step, TEAM_OF_SEAT
from obs.encoder import encode_state, update_void_flags
from cards import id_to_card
from utils.HeuristicAgent import HeuristicAgent
from train_dqn import DQNNet, feature_dim, DEVICE

# ================================
# CONFIG
# ================================
PLAYERS = ["dqn", "heuristic", "dqn", "heuristic"]  # setup dei 4 giocatori
CKPT = "dqn_tressette_checkpoint_ep690000.pt"  # checkpoint da usare
N_MATCHES = 16384  # quante partite in totale
BATCH_SIZE = 32  # quante partite simultanee

"Faciamo tutti i forward pass in parallelo, creazione ambiente, step e calcolo risultati sempre su cpu"

# ================================
# MODELLO
# ================================
policy = DQNNet(feature_dim()).to(DEVICE)
checkpoint = torch.load(CKPT, map_location=DEVICE)
policy.load_state_dict(checkpoint["model"])
policy.eval()

print(f"✅ Modello caricato da {CKPT}")
print(f"💻 Device attivo: {DEVICE}")

# ================================
# FUNZIONI
# ================================
def choose_action_batch(states, void_flags_batch):
    "Ci serve per gestire le scelte di più partite in contemporanea"
    "Calcola le azioni per un batch di stati (tutti seat correnti)"
    x_list, mask_list = [], []
    for s, void_flags in zip(states, void_flags_batch):
        seat = s.current_player
        x, mask = encode_state(s, seat, void_flags)
        x_list.append(x)
        mask_list.append(mask)
    x = torch.cat(x_list, dim=0).to(DEVICE)
    mask = torch.cat(mask_list, dim=0).to(DEVICE)

    "Calcoliamo tutti i Q-Values in parallelo"
    with torch.no_grad():
        q = policy(x, mask)
        q = q.masked_fill(mask == 0, -1e9)
        actions = torch.argmax(q, dim=1).tolist()
    return actions


"Versione play_one_game adatTata per GPU, utilizziamo batch per gestire più partite in contemporanea"
def play_many_games(n_matches=N_MATCHES, batch_size=BATCH_SIZE):
    wins_team0 = wins_team1 = draws = 0
    all_seeds = [int(time.time() * 1e6) % (2**32 - 1) + i for i in range(n_matches)]

    for start_idx in tqdm(range(0, n_matches, batch_size), desc="🎮 Playing"):
        batch_seeds = all_seeds[start_idx:start_idx + batch_size]
        states = [deal(leader=random.randint(0, 3), rng=random.Random(s)) for s in batch_seeds]
        void_flags_batch = [[[0]*4 for _ in range(4)] for _ in states]
        done_flags = [False] * len(states)
        rewards_batch = [None] * len(states)

        while not all(done_flags):
            active_idx = [i for i, d in enumerate(done_flags) if not d]
            current_states = [states[i] for i in active_idx]
            current_voids = [void_flags_batch[i] for i in active_idx]
            actions = choose_action_batch(current_states, current_voids)

            for idx, act in zip(active_idx, actions):
                seat = states[idx].current_player
                update_void_flags(void_flags_batch[idx], states[idx], seat, act)
                states[idx], rewards, done, _ = step(states[idx], act)
                if done:
                    rewards_batch[idx] = rewards
                    done_flags[idx] = True

        for rewards in rewards_batch:
            if rewards is None:
                continue
            if int(rewards[0]) > int(rewards[1]):
                wins_team0 += 1
            elif int(rewards[1]) > int(rewards[0]):
                wins_team1 += 1
            else:
                draws += 1

    print("\n=== RISULTATI FINALI ===")
    print(f"Team0 (seat 0+2: {PLAYERS[0]} + {PLAYERS[2]}) → {wins_team0}/{n_matches}")
    print(f"Team1 (seat 1+3: {PLAYERS[1]} + {PLAYERS[3]}) → {wins_team1}/{n_matches}")



if __name__ == "__main__":
    print("=== TORNEO PARALLELO TRESSETTE ===")
    start = time.time()
    play_many_games()
    print(f"\n⏱️ Tempo totale: {time.time() - start:.2f} secondi")