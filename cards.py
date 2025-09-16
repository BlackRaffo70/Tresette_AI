from __future__ import annotations
from dataclasses import dataclass
from typing import List

# Semi (italiani): 0=Denari (D), 1=Coppe (C), 2=Spade (S), 3=Bastoni (B)
# Nota: le sigle D/C/S/B sono solo per stampa compatta.
SUITS = ["D", "C", "S", "B"]

# Ranks per visualizzazione (dal più basso al più alto a fini estetici)
RANKS = ["4", "5", "6", "7", "8", "9", "10", "A", "2", "3"]

# Ordine di forza del Tresette (dal più forte al più debole):
# 3 > 2 > A > R > C > F > 7 > 6 > 5 > 4
STRENGTH_DESC = ["3", "2", "A", "10", "9", "8", "7", "6", "5", "4"]

# Mappiamo ogni rank a un intero di forza (9=massima, 0=minima)
RANK_TO_STRENGTH = {r: 9 - i for i, r in enumerate(STRENGTH_DESC)}
STRENGTH_TO_RANK = {v: k for k, v in RANK_TO_STRENGTH.items()}

# Punteggio in terzi di punto (evitiamo i float):
# Asso=3 terzi (=1 punto intero), 2/3/Re/Cavallo/Fante=1 terzo, le altre=0
THIRDS_BY_RANK = {
    "A": 3,
    "2": 1,
    "3": 1,
    "10": 1,
    "9": 1,
    "8": 1,
    "7": 0,
    "6": 0,
    "5": 0,
    "4": 0,
}

@dataclass(frozen=True)
class Card:
    """Rappresenta una carta del mazzo italiano (40 carte)."""
    suit: int  # 0..3
    rank: str  # una tra RANKS

    @property
    def strength(self) -> int:
        """Valore di forza del rank secondo l'ordine del Tresette."""
        return RANK_TO_STRENGTH[self.rank]

    @property
    def thirds(self) -> int:
        """Ritorna il valore della carta in terzi di punto."""
        return THIRDS_BY_RANK[self.rank]

    def __str__(self) -> str:
        return f"{self.rank}{SUITS[self.suit]}"

def id_to_card(cid: int) -> Card:
    """Converte un id (0..39) nella corrispondente Card."""
    suit = cid // 10
    rix = cid % 10
    return Card(suit, RANKS[rix])

def card_to_id(card: Card) -> int:
    """Converte una Card nel suo id (0..39)."""
    return card.suit * 10 + RANKS.index(card.rank)

def full_deck() -> List[int]:
    """Ritorna il mazzo completo (lista di id 0..39)."""
    return [s * 10 + r for s in range(4) for r in range(10)]
