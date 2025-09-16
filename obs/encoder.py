# ================================
# File: obs/encoder.py
# ================================
import torch
from game4p import GameState, legal_action_mask
from cards import id_to_card

def update_void_flags(void_flags, trick):
    """Aggiorna i void flags: se un giocatore non segue il seme d'uscita, segna che è void su quel seme."""
    if len(trick.plays) < 2:
        return void_flags  # niente da aggiornare se non ci sono abbastanza carte
    lead_suit = id_to_card(trick.plays[0][1]).suit
    for seat, cid in trick.plays[1:]:
        suit = id_to_card(cid).suit
        if suit != lead_suit:
            void_flags[seat][lead_suit] = 1
    return void_flags

def encode_state(state: GameState, seat: int, void_flags):
    """Trasforma lo stato in un vettore di feature PyTorch + mask delle azioni legali."""
    # One-hot della mano (40)
    hand_vec = torch.zeros(40)
    for c in state.hands[seat]:
        hand_vec[c] = 1.0

    # Carte già uscite (40)
    played_vec = torch.zeros(40)
    # carte giocate nei trick
    for _, cid in state.trick.plays:
        played_vec[cid] = 1.0
    # carte catturate
    for team_cards in state.captures_team.values():
        for c in team_cards:
            played_vec[c] = 1.0

    # Flatten dei void flags (4x4)
    voids = torch.tensor(void_flags, dtype=torch.float32).flatten()

    # Seat e compagno
    seat_id = torch.zeros(4); seat_id[seat] = 1.0
    ally_pos = torch.zeros(4); ally_pos[(seat+2) % 4] = 1.0

    # Concatenazione feature
    features = torch.cat([hand_vec, played_vec, voids, seat_id, ally_pos])
    features = features.unsqueeze(0)  # shape [1, dim]

    # Mask azioni legali
    mask = torch.tensor(legal_action_mask(state), dtype=torch.float32).unsqueeze(0)

    return features, mask

# One-hot segnali: 4 giocatori × 4 semi × 3 segnali
signals_tensor = torch.zeros(4, 4, 3)
for seat, sig in state.signals.items():
    seme = sig["suit"]
    if sig["signal"] == "volo":
        signals_tensor[seat, seme, 0] = 1
    elif sig["signal"] == "striscio":
        signals_tensor[seat, seme, 1] = 1
    elif sig["signal"] == "busso":
        signals_tensor[seat, seme, 2] = 1
