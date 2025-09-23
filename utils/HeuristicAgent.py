from cards import id_to_card, RANK_TO_STRENGTH

class HeuristicAgent:
    def __init__(self, name="Heuristic"):
        self.name = name

    @staticmethod
    def choose_action(state, legal_idx):
        """Sceglie un'azione euristica dato lo stato e gli id legali delle carte."""
        # Converti gli id legali in oggetti Card
        legal_cards = [id_to_card(cid) for cid in legal_idx]

        # =====================
        # 1. Assecondare dichiarazioni del compagno
        # =====================
        for seat, sig in state.signals.items():
            seme = sig["suit"]
            if sig["signal"] == "busso" or sig["signal"] == "striscio":
                for cid, card in zip(legal_idx, legal_cards):
                    if card.suit == seme:
                        return cid
            elif sig["signal"] == "volo":
                # segnala di voler prendere quel seme, non giocarlo
                continue

        # =====================
        # 2. Palo più forte con gestione dell'asso
        # =====================
        suits = [card.suit for card in legal_cards]
        ranks = [card.rank for card in legal_cards]

        # Palo più forte = somma dei valori delle carte legali
        strongest_suit = max(set(suits), key=lambda suit: sum(RANK_TO_STRENGTH[card.rank]
                                                              for card in legal_cards if card.suit == suit))
        for cid, card in zip(legal_idx, legal_cards):
            if card.suit != strongest_suit:
                continue
            # Gestione asso: giocalo solo se hai 2 e 3 in mano o sono già usciti
            hand_ranks = [id_to_card(c).rank for c in state.hands[state.current_player]]
            table_ranks = [id_to_card(c).rank for c in sum(state.hands, []) if c not in sum(state.hands, [])]  # opzionale
            if card.rank == 'A':
                if '2' in hand_ranks and '3' in hand_ranks:
                    return cid
                else:
                    continue
            else:
                return cid

        # =====================
        # 3. Giocare 2 se hai esattamente due carte di quel seme
        # =====================
        for cid, card in zip(legal_idx, legal_cards):
            same_suit_cards = [c for c in legal_cards if c.suit == card.suit]
            if card.rank == '2' and len(same_suit_cards) == 2:
                return cid

        # =====================
        # 4. Cambiare sempre gioco se nessun segnale
        # =====================
        played_suits = [id_to_card(c).suit for c in sum(state.hands, []) if c not in sum(state.hands, [])]  # opzionale
        for cid, card in zip(legal_idx, legal_cards):
            if card.suit not in played_suits:
                return cid

        # =====================
        # 5. Assicurarsi l'ultima presa (gioca carta più alta)
        # =====================
        return max(legal_idx, key=lambda cid: RANK_TO_STRENGTH[id_to_card(cid).rank])
