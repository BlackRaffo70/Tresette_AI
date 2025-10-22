# ================================
# File: menu_cli.py
# ================================
from __future__ import annotations
import random
import torch
from cards import id_to_card, SUITS
from game4p import deal, step, legal_action_mask
from obs.encoder import encode_state
from utils.HeuristicAgent import HeuristicAgent
from train_dqn import DQNNet, feature_dim, DEVICE  # import per DQN

def print_hand(seat: int, hand: list[int]):
    """Stampa la mano di un giocatore (solo umano)."""
    print(f"\nTua mano (Seat {seat}):")
    for idx, cid in enumerate(hand):
        print(f"  {idx}: {id_to_card(cid)}")

# ================================
# HUMAN vs AI (Random, Heuristic o DQN)
# ================================

"Caricare il modello pt da utilizzare"
dqn_model = "dqn_tressette_checkpoint_ep690000.pt"
def play_human_vs_random(seed: int = 0, human_seat: int = 0):
    #Genero il mazzo e assegno le carte tramite funzione deal di cards
    rng = random.Random(seed)
    st = deal(rng=rng, leader=0)

    print("=== Inizio mano di Tressette (4 giocatori) ===")
    print(f"Tu sei Seat {human_seat}. Il tuo compagno è Seat {(human_seat+2)%4}.\n")

    while True:
        current = st.current_player
        mask = legal_action_mask(st)

        # Turno umano
        if current == human_seat:
            print_hand(human_seat, st.hands[human_seat])
            legali = [cid for cid in st.hands[human_seat] if mask[cid] == 1]
            print("\nCarte legali:")
            for idx, cid in enumerate(legali):
                print(f"  {idx}: {id_to_card(cid)}")
            scelta = int(input("Scegli l'indice della carta da giocare: "))
            cid = legali[scelta]

        # Turno AI random
        else:
            legali = [cid for cid in st.hands[current] if mask[cid] == 1]
            cid = rng.choice(legali)

        prev_player = st.current_player
        st, rew, done, info = step(st, cid)
        print(f"\nSeat {prev_player} gioca {id_to_card(cid)}")

        # Se c'è stato un segnale (leader del trick)
        if "signal" in info:
            sig = info["signal"]
            seme_str = SUITS[sig["suit"]]
            print(f"  >>> Segnale Seat {sig['seat']}: {sig['signal'].upper()} su seme {seme_str}")

        # Se la presa si è appena chiusa
        if len(st.trick.plays) == 0:
            print(f"  -> Presa vinta dal Seat {st.current_player}\n")

        if done:
            print("=== Mano terminata ===")
            print("Punteggio finale:", info["points"])
            print("Team 0 (seats 0 e 2):", info["points"][0], "punti")
            print("Team 1 (seats 1 e 3):", info["points"][1], "punti")
            break

def play_human_vs_ai(mode: str = "random", ckpt_path: str | None = None, seed: int = 0, human_seat: int = 0):
    """Permette di giocare contro 3 AI: random, heuristic o DQN."""
    rng = random.Random(seed)
    st = deal(rng=rng, leader=0)

    print(f"=== Inizio mano di Tressette (modalità: {mode.upper()}) ===")
    print(f"Tu sei Seat {human_seat}. Il tuo compagno è Seat {(human_seat+2)%4}.\n")

    dqn_policy = None
    if mode == "dqn" and ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        dqn_policy = DQNNet(feature_dim()).to(DEVICE)
        dqn_policy.load_state_dict(ckpt["model"])
        dqn_policy.eval()
        print(f"✅ Modello DQN caricato da {ckpt_path}")

    void_flags = [[0]*4 for _ in range(4)]

    while True:
        current = st.current_player
        mask = legal_action_mask(st)

        # Turno umano
        if current == human_seat:
            print_hand(human_seat, st.hands[human_seat])
            legali = [cid for cid in st.hands[human_seat] if mask[cid] == 1]
            print("\nCarte legali:")
            for idx, cid in enumerate(legali):
                print(f"  {idx}: {id_to_card(cid)}")
            scelta = int(input("Scegli l'indice della carta da giocare: "))
            cid = legali[scelta]

        # Turno AI
        else:
            legali = [cid for cid in st.hands[current] if mask[cid] == 1]

            if mode == "random":
                cid = rng.choice(legali)
            elif mode == "heuristic":
                cid = HeuristicAgent.choose_action(st, legali)
            elif mode == "dqn" and dqn_policy is not None:
                x, m = encode_state(st, current, void_flags)
                x, m = x.to(DEVICE), m.to(DEVICE)
                with torch.no_grad():
                    q = dqn_policy(x, m).squeeze(0)
                    q_legali = q[legali]
                    cid = legali[torch.argmax(q_legali).item()]

        prev_player = st.current_player
        st, rew, done, info = step(st, cid)
        print(f"\nSeat {prev_player} gioca {id_to_card(cid)}")

        # Se c'è stato un segnale (leader del trick)
        if "signal" in info:
            sig = info["signal"]
            seme_str = SUITS[sig["suit"]]
            print(f"  >>> Segnale Seat {sig['seat']}: {sig['signal'].upper()} su seme {seme_str}")

        # Se la presa si è appena chiusa
        if len(st.trick.plays) == 0:
            print(f"  -> Presa vinta dal Seat {st.current_player}\n")

        if done:
            print("=== Mano terminata ===")
            print("Punteggio finale:", info["points"])
            print("Team 0 (seats 0 e 2):", info["points"][0], "punti")
            print("Team 1 (seats 1 e 3):", info["points"][1], "punti")
            break

def main_menu():
    while True:
        print("\n=== Menu Tressette 4P ===")
        print("1) Gioca una mano (tu vs 3 AI random)")
        print("2) Gioca contro AI euristica")
        print("3) Gioca contro AI DQN")
        print("0) Esci")
        scelta = input("Seleziona opzione: ")

        if scelta == "1":
            seat = int(input("Scegli il tuo seat (0-3): "))
            play_human_vs_random(human_seat=seat)
        elif scelta == "2":
            seat = int(input("Scegli il tuo seat (0-3): "))
            play_human_vs_ai(mode="heuristic", human_seat=seat)
        elif scelta == "3":
            seat = int(input("Scegli il tuo seat (0-3): "))
            play_human_vs_ai(mode="dqn", ckpt_path=dqn_model, human_seat=seat)
        elif scelta == "0":
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida, riprova.")

if __name__ == "__main__":
    main_menu()