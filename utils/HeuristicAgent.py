from cards import id_to_card, RANK_TO_STRENGTH

class HeuristicAgent:
    def __init__(self, name="Heuristic"):
        self.name = name

    @staticmethod
    def choose_action(state, legal_idx):
        """Sceglie un'azione euristica dato lo stato e gli id legali delle carte."""
        legal_cards = [id_to_card(cid) for cid in legal_idx]

        # =====================
        # 1. Assecondare dichiarazioni del compagno
        # =====================
        for seat_signal, sig in state.signals.items():
            suit_signal = sig["suit"]
            signal_type = sig["signal"]
            if signal_type in ("busso", "striscio"):
                for cid_signal, card_signal in zip(legal_idx, legal_cards):
                    if card_signal.suit == suit_signal:
                        return cid_signal
            elif signal_type == "volo":
                continue

        # =====================
        # 2. Palo più forte con gestione dell'asso
        # =====================
        suits_in_hand = [card_suit.suit for card_suit in legal_cards]
        strongest_suit = max(
            set(suits_in_hand),
            key=lambda s: sum(RANK_TO_STRENGTH[card_suit.rank]
                              for card_suit in legal_cards if card_suit.suit == s)
        )
        for cid_strong, card_strong in zip(legal_idx, legal_cards):
            if card_strong.suit != strongest_suit:
                continue
            hand_ranks_strong = [id_to_card(cid_hand).rank for cid_hand in state.hands[state.current_player]]
            if card_strong.rank == 'A':
                if '2' in hand_ranks_strong and '3' in hand_ranks_strong:
                    return cid_strong
                else:
                    continue
            else:
                return cid_strong

        # =====================
        # 3. Giocare 2 se hai esattamente due carte di quel seme
        # =====================
        for cid_two, card_two in zip(legal_idx, legal_cards):
            same_suit_cards_two = [c for c in legal_cards if c.suit == card_two.suit]
            if card_two.rank == '2' and len(same_suit_cards_two) == 2:
                return cid_two

        # =====================
        # 4. Cambiare sempre gioco se nessun segnale
        # =====================
        for cid_change, card_change in zip(legal_idx, legal_cards):
            return cid_change  # fallback semplice

        # =====================
        # 5. Assicurarsi l'ultima presa (gioca carta più alta)
        # =====================
        return max(legal_idx, key=lambda cid_max: RANK_TO_STRENGTH[id_to_card(cid_max).rank])
