# ================================
# File: menu_cli.py
# ================================
from __future__ import annotations
import random
from cards import id_to_card
from game4p import deal, step, legal_action_mask

def print_hand(seat: int, hand: list[int]):
    """Stampa la mano di un giocatore (usato per l'umano)."""
    print(f"Tua mano (Seat {seat}):")
    for idx, cid in enumerate(hand):
        print(f"  {idx}: {id_to_card(cid)}")

def play_human_vs_random(seed: int = 0, human_seat: int = 0):
    rng = random.Random(seed)
    st = deal(rng=rng, leader=0)

    print("=== Inizio mano di Tresette (4 giocatori) ===")
    print(f"Tu sei il Seat {human_seat}. Il tuo compagno è Seat {(human_seat+2)%4}.\n")

    while True:
        current = st.current_player
        mask = legal_action_mask(st)

        if current == human_seat:
            # Turno umano
            print_hand(human_seat, st.hands[human_seat])
            # Filtra le carte legali
            legali = [cid for cid in st.hands[human_seat] if mask[cid] == 1]
            print("\nCarte legali:")
            for idx, cid in enumerate(legali):
                print(f"  {idx}: {id_to_card(cid)}")
            # Input utente
            scelta = int(input("Scegli l'indice della carta da giocare: "))
            cid = legali[scelta]
        else:
            # Turno AI random
            legali = [cid for cid in st.hands[current] if mask[cid] == 1]
            cid = rng.choice(legali)

        # Esegui la mossa
        prev_player = current
        st, rew, done, info = step(st, cid)
        print(f"Seat {prev_player} gioca {id_to_card(cid)}")

        # Se la presa è stata completata
        if len(st.trick.plays) == 0:
            print(f"  -> Presa vinta dal seat {st.current_player}\n")

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
        print(f"Seat {prev_player} gioca {id_to_card(cid)}")

        if len(st.trick.plays) == 0:
            print(f"  -> Presa vinta dal seat {st.current_player}\n")

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
            play_human_vs_random()
        elif scelta == "2":
            play_full_random()
        elif scelta == "0":
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida, riprova.")

if __name__ == "__main__":
    main_menu()
