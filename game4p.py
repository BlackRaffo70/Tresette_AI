# ================================
# File: game4p.py
# ================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random

from cards import full_deck
from rules import Trick, legal_actions, score_team_from_captures, TEAM_OF_SEAT

@dataclass
class GameState:
    """Stato di una mano a 4 giocatori (2 coppie).

    Attributi:
      - hands: lista di 4 mani (ognuna lista di id carta)
      - current_player: seat (0..3) a cui tocca giocare
      - trick: presa corrente (può essere vuota o in corso)
      - captures_team: mapping team -> carte catturate
      - last_trick_winner: seat che ha vinto l'ultima presa completata
      - tricks_played: numero di prese completate (0..10)
    """
    hands: List[List[int]]
    current_player: int
    trick: Trick
    captures_team: Dict[int, List[int]]
    last_trick_winner: int | None
    tricks_played: int

    def clone(self) -> "GameState":
        """Crea una copia indipendente dello stato."""
        return GameState(
            hands=[h.copy() for h in self.hands],
            current_player=self.current_player,
            trick=Trick(self.trick.leader, self.trick.plays.copy()),
            captures_team={k: v.copy() for k, v in self.captures_team.items()},
            last_trick_winner=self.last_trick_winner,
            tricks_played=self.tricks_played,
        )

def deal(rng: random.Random | None = None, leader: int = 0) -> GameState:
    """Distribuisce il mazzo (10 carte a testa) e inizializza lo stato.

    Parametri:
      - rng: generatore random (opzionale)
      - leader: seat che inizia la prima presa
    """
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
    )

def step(state: GameState, card_id: int) -> Tuple[GameState, Dict[int,int], bool, Dict]:
    """Esegue una giocata dal giocatore corrente.

    Parametri:
      - state: stato attuale della mano
      - card_id: id della carta giocata

    Ritorna:
      - nuovo stato
      - reward per team (0 finché la mano non finisce)
      - done=True se la mano è terminata
      - info: dettagli aggiuntivi (punti finali, carte catturate)
    """
    p = state.current_player
    assert card_id in state.hands[p], "Carta non in mano"
    la = legal_actions(state.hands[p], state.trick)
    assert card_id in la, "Mossa illegale: devi seguire il seme se puoi"

    ns = state.clone()
    # Rimuovi la carta giocata
    ns.hands[p].remove(card_id)
    ns.trick.plays.append((p, card_id))

    rewards = {0: 0, 1: 0}
    info = {}

    # Se la presa non è ancora completa, passa al prossimo giocatore
    if len(ns.trick.plays) < 4:
        ns.current_player = (p + 1) % 4
        done = False
        return ns, rewards, done, info

    # Altrimenti la presa è completa → determina vincitore
    winner = ns.trick.winner()
    ns.last_trick_winner = winner
    taken = [cid for _, cid in ns.trick.plays]
    team = TEAM_OF_SEAT[winner]
    ns.captures_team[team].extend(taken)
    ns.tricks_played += 1

    # Resetta la presa e assegna il turno al vincitore
    ns.trick = Trick(winner, [])
    ns.current_player = winner

    # Mano terminata dopo 10 prese?
    if ns.tricks_played == 10:
        last_team = TEAM_OF_SEAT[ns.last_trick_winner] if ns.last_trick_winner is not None else None
        team0_last = (last_team == 0)
        team1_last = (last_team == 1)
        t0 = score_team_from_captures(ns.captures_team[0], team0_last)
        t1 = score_team_from_captures(ns.captures_team[1], team1_last)
        rewards = {0: t0, 1: t1}
        done = True
        info = {"points": rewards, "captures": ns.captures_team}
    else:
        done = False

    return ns, rewards, done, info

def legal_action_mask(state: GameState) -> List[int]:
    """Ritorna una maschera binaria (40 elementi).
    1 = carta giocabile dal giocatore corrente, 0 = non giocabile.
    """
    la = set(legal_actions(state.hands[state.current_player], state.trick))
    mask = [1 if cid in la else 0 for cid in range(40)]
    return mask
