# ================================
# File: menu_cli.py
# ================================
from __future__ import annotations
import random
from cards import id_to_card, SUITS
from game4p import deal, step, legal_action_mask

def print_hand(seat: int, hand: list[int]):
    """Stampa la mano di un giocatore (solo umano)."""
    print(f"\nTua mano (Seat {seat}):")
    for idx, cid in enumerate(hand):
        print(f"  {idx}: {id_to_card(cid)}")

def play_human_vs_random(seed: int = 0, human_seat: int = 0):
    rng = random.Random(seed)
    st = deal(rng=rng, leader=0)

    print("=== Inizio mano di Tresette (4 giocatori) ===")
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

def play_full_random(seed: int = 0):
    rng = random.Random(seed)
    st = deal(rng=rng, leader=0)

    print("=== Mano simulata con 4 AI random ===\n")

    while True:
        mask = legal_action_mask(st)
        legali = [cid for cid in st.hands[st.current_player] if mask[cid] == 1]
        cid = rng.choice(legali)

        prev_player = st.current_player
        st, rew, done, info = step(st, cid)
        print(f"\nSeat {prev_player} gioca {id_to_card(cid)}")

        if "signal" in info:
            sig = info["signal"]
            seme_str = SUITS[sig["suit"]]
            print(f"  >>> Segnale Seat {sig['seat']}: {sig['signal'].upper()} su seme {seme_str}")

        if len(st.trick.plays) == 0:
            print(f"  -> Presa vinta dal Seat {st.current_player}\n")

        if done:
            print("=== Mano terminata ===")
            print("Punteggio finale:", info["points"])
            break

def main_menu():
    while True:
        print("\n=== Menu Tresette 4P ===")
        print("1) Gioca una mano (tu vs 3 AI random)")
        print("2) Simula una mano (4 AI random)")
        print("0) Esci")
        scelta = input("Seleziona opzione: ")

        if scelta == "1":
            seat = int(input("Scegli il tuo seat (0-3): "))
            play_human_vs_random(human_seat=seat)
        elif scelta == "2":
            play_full_random()
        elif scelta == "0":
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida, riprova.")

if __name__ == "__main__":
    main_menu()
