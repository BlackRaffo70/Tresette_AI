# ================================
# File: rules.py
# ================================
from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass

from cards import id_to_card

# Mappatura dei posti (seats) nelle squadre:
# team 0 = seats 0 e 2 ; team 1 = seats 1 e 3
TEAM_OF_SEAT = {0: 0, 2: 0, 1: 1, 3: 1}

@dataclass
class Trick:
    """Rappresenta una presa (trick), parziale o completa.
    ""TRICK =1 GIRO DI GIOCATE -> NON 1 PARTITA""

    Attributi:
      - leader: seat che ha aperto la presa, ultimo che prende
      - plays: lista di tuple (seat, card_id) in ordine di gioco
    """
    leader: int
    plays: List[Tuple[int, int]]

    def lead_suit(self) -> int:
        """Seme d'uscita della presa (quello della prima carta giocata)."""
        if not self.plays:
            return -1
        return id_to_card(self.plays[0][1]).suit

    def is_complete(self) -> bool:
        """True se nella presa sono state giocate 4 carte."""
        return len(self.plays) == 4

    def winner(self) -> int:
        """Ritorna il seat del giocatore che ha vinto la presa.
        Vince la carta più forte del seme di uscita.
        """
        assert self.is_complete(), "Trick incompleto"

        """assert = verifica automatica (una specie di “freno di emergenza”) che serve a controllare se una condizione è vera —
e se non lo è, il programma si ferma subito e genera un’eccezione """

        suit = self.lead_suit()
        best_seat, best_card = self.plays[0]
        best_strength = id_to_card(best_card).strength
        for seat, cid in self.plays[1:]:
            c = id_to_card(cid)
            if c.suit == suit and c.strength > best_strength:
                best_strength = c.strength
                best_seat = seat
        return best_seat

def legal_actions(hand: List[int], trick: Trick) -> List[int]:
    """Ritorna le carte legali giocabili dalla mano.
    - Se non c'è ancora una carta nel trick: qualsiasi carta è valida.
    - Altrimenti, se ho almeno una carta del seme di uscita: devo seguirlo.
    - Se non ne ho: posso giocare qualsiasi carta.
    """
    if not trick.plays:
        return sorted(hand)
    lead = trick.lead_suit()
    same_suit = [cid for cid in hand if id_to_card(cid).suit == lead]
    return sorted(same_suit if same_suit else hand)

def score_cards_thirds(cards: List[int]) -> int:
    """Somma i terzi di punto delle carte (A=3, 2/3/R/C/F=1, altre=0)."""
    return sum(id_to_card(cid).thirds for cid in cards)

def score_team_from_captures(captured_cards: List[int], took_last_trick: bool) -> int:
    """Calcola i punti di una squadra dalle carte catturate.

    Regole:
    - ogni Asso vale 1 punto, 2/3/Re/Cavallo/Fante valgono 1/3
    - si arrotonda per difetto ai punti interi
    - +1 punto bonus a chi prende l’ultima presa
    """
    thirds = score_cards_thirds(captured_cards)
    pts = thirds // 3  # arrotondamento per difetto
    if took_last_trick:
        pts += 1
    return int(pts)
