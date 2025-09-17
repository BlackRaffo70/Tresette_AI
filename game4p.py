# ================================
# File: game4p.py
# ================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

from cards import full_deck, id_to_card
from rules import Trick, legal_actions, score_team_from_captures, TEAM_OF_SEAT

@dataclass
class GameState: #rappresenta la mano attuale
    """Stato di una mano a 4 giocatori (2 coppie)."""
    hands: List[List[int]]
    current_player: int
    trick: Trick
    captures_team: Dict[int, List[int]]
    last_trick_winner: int | None
    tricks_played: int
    signals: Dict[int, Dict]  # seat -> {suit, signal}

    def clone(self) -> "GameState": #crea uno stato clone indipendente da quello originale
        """Crea una copia indipendente dello stato."""
        return GameState(
            hands=[h.copy() for h in self.hands],
            current_player=self.current_player,
            trick=Trick(self.trick.leader, self.trick.plays.copy()),
            captures_team={k: v.copy() for k, v in self.captures_team.items()},
            last_trick_winner=self.last_trick_winner,
            tricks_played=self.tricks_played,
            signals={k: v.copy() for k, v in self.signals.items()},
        )

def deal(rng: random.Random | None = None, leader: int = 0) -> GameState:
    """Distribuisce il mazzo (10 carte a testa) e inizializza lo stato."""
    rng = rng or random.Random()
    deck = full_deck()
    rng.shuffle(deck)
    hands = [sorted(deck[i*10:(i+1)*10]) for i in range(4)]
    return GameState(
        hands=hands,
        current_player=leader,
        trick=Trick(leader, []),
        captures_team={0: [], 1: []},
        last_trick_winner=None,
        tricks_played=0,
        signals={}
    )

def step(state: GameState, card_id: int) -> Tuple[GameState, Dict[int,int], bool, Dict]:
    """Esegue una giocata dal giocatore corrente e calcola eventuali segnali."""
    p = state.current_player
    assert card_id in state.hands[p], "Carta non in mano"
    la = legal_actions(state.hands[p], state.trick)
    assert card_id in la, "Mossa illegale: devi seguire il seme se puoi"

    ns = state.clone() #Ns-> State -> Gamestate -> mano di un giocatore
    ns.hands[p].remove(card_id)
    ns.trick.plays.append((p, card_id))

    rewards = {0: 0, 1: 0}
    info = {}

    # Se è il leader del trick → comunica il segnale
    if len(ns.trick.plays) == 1:
        played_card = id_to_card(card_id)
        seme = played_card.suit
        rest = [c for c in ns.hands[p] if id_to_card(c).suit == seme]

        # logica aggiornata
        if any(id_to_card(c).rank in ["2", "3"] for c in rest):
            signal = "busso"
        elif not rest:
            signal = "volo"
        else:
            signal = "striscio"

        ns.signals[p] = {"suit": seme, "signal": signal}
        info["signal"] = {"seat": p, "suit": seme, "signal": signal}

    # Se la presa non è ancora completa
    if len(ns.trick.plays) < 4:
        ns.current_player = (p + 1) % 4
        return ns, rewards, False, info

    # Altrimenti la presa è completa → determina vincitore
    winner = ns.trick.winner()
    ns.last_trick_winner = winner
    taken = [cid for _, cid in ns.trick.plays]
    team = TEAM_OF_SEAT[winner]
    ns.captures_team[team].extend(taken)
    ns.tricks_played += 1

    ns.trick = Trick(winner, [])
    ns.current_player = winner

    if ns.tricks_played == 10:
        last_team = TEAM_OF_SEAT[ns.last_trick_winner] if ns.last_trick_winner is not None else None
        t0 = score_team_from_captures(ns.captures_team[0], last_team == 0)
        t1 = score_team_from_captures(ns.captures_team[1], last_team == 1)
        rewards = {0: t0, 1: t1}
        info.update({"points": rewards, "captures": ns.captures_team, "signals": ns.signals})
        return ns, rewards, True, info

    return ns, rewards, False, info

def legal_action_mask(state: GameState) -> List[int]:
    """Ritorna una maschera binaria (40 elementi)."""
    la = set(legal_actions(state.hands[state.current_player], state.trick))
    return [1 if cid in la else 0 for cid in range(40)]
