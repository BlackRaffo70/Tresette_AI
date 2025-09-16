# ================================
# File: tests/test_rules.py
# ================================
import pytest
from cards import Card, card_to_id
from rules import Trick, legal_actions

# Funzione di utilità: crea una mano a partire da oggetti Card
def make_hand(cards):
    return [card_to_id(c) for c in cards]

def test_follow_suit_enforced():
    # Il primo giocatore gioca 7 di Coppe → seme di uscita = 1
    trick = Trick(leader=0, plays=[(0, card_to_id(Card(1, '7')))])

    # Mano del secondo giocatore: A di Coppe, A di Spade, 3 di Bastoni
    hand = make_hand([Card(1, 'A'), Card(2, 'A'), Card(3, '3')])

    la = legal_actions(hand, trick)

    # Deve essere legale solo A di Coppe (obbligo di seguire il seme)
    assert set(la) == {card_to_id(Card(1, 'A'))}

def test_trick_winner_simple():
    # Seme di uscita: Spade (2)
    # Vince la carta più forte di Spade, in questo caso 3♠
    t = Trick(leader=1, plays=[
        (1, card_to_id(Card(2, '7'))),   # 7 di Spade
        (2, card_to_id(Card(2, 'A'))),   # A di Spade
        (3, card_to_id(Card(2, '3'))),   # 3 di Spade (più forte di tutti)
        (0, card_to_id(Card(0, '3'))),   # 3 di Denari (altro seme → ignorato)
    ])

    assert t.is_complete()
    assert t.winner() == 3  # vince il giocatore 3 con 3♠
