# ================================
# File: demo_cli.py
# ================================
from __future__ import annotations
import random
from cards import id_to_card
from game4p import deal, step, legal_action_mask

# Dimostrazione: 4 agenti random giocano una mano completa
# e vengono stampati i dettagli delle prese e i punteggi finali.

def play_random_hand(seed: int = 0):
    rng = random.Random(seed)
    st = deal(rng=rng, leader=0)

    # Stampa le mani iniziali
    print("Mani iniziali:")
    for s in range(4):
        print(f"Seat {s}:", " ".join(str(id_to_card(c)) for c in st.hands[s]))
    print("\nGioco in corso...\n")

    last_player = None
    while True:
        # Ottieni la maschera delle azioni legali
        mask = legal_action_mask(st)
        # Seleziona una carta legale a caso dalla mano del giocatore corrente
        legal = [i for i, m in enumerate(mask) if m == 1 and i in st.hands[st.current_player]]
        cid = rng.choice(legal)

        last_player = st.current_player
        st, rew, done, info = step(st, cid)

        print(f"Seat {last_player} gioca {id_to_card(cid)}")

        # Se la presa si è appena chiusa
        if len(st.trick.plays) == 0:
            print(f"  -> Presa vinta dal seat {st.current_player}\n")

        # Se la mano è finita, stampa il risultato
        if done:
            print("Mano terminata. Punti:", info["points"])
            print("Team 0 carte catturate:", len(info["captures"][0]))
            print("Team 1 carte catturate:", len(info["captures"][1]))
            break

# Avvio demo se eseguito da riga di comando
if __name__ == "__main__":
    play_random_hand(42)
