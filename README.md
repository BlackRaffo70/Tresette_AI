# 🃏 Tresette AI

La nostra attività progettuale per il corso di Fondamenti di Intelligenza Artificiale si è basato sulla realizzazione di un sistema per la progettazione, l’allenamento e la valutazione di un’Intelligenza Artificiale capace di sfidare gli utenti al gioco di carte **Tresette**, tutto ciò applicando tecniche di **Reinforcement Learning (RL)** e simulazioni di partite.

---

## 🎯 Obiettivi

- **Modellare** le regole e le dinamiche del gioco del Tresette. -> Essere in grado di giocare contro un'AI non allenata(random)
- **Definire** un ambiente simulato in cui un agente AI possa giocare e migliorare.  -> Effettuare un training
- **Addestrare** l’agente tramite algoritmi di *Deep Reinforcement Learning* (es. Deep Q-Learning).  
- **Permettere** a un giocatore umano di sfidare l’AI.  
- **Confrontare** strategie casuali, euristiche e di apprendimento.  

---

## 🃏 Regole implementate

- Gestione del **mazzo** e distribuzione delle carte.  
- Turni di gioco per ** 4 giocatori** attraverso lista di mosse legali.  
- Calcolo delle **prese** e dei **punteggi** secondo le regole del Tresette. -> 1/3 per le figure , 1 per l'asso
- Definizione delle **condizioni di vittoria**.  

---

## 🧠 Architettura AI

- **Ambiente** → rappresenta lo stato della partita (mani, prese, turno, carte giocate).  
- **Agente** → apprende a selezionare le mosse tramite `ε-greedy policy`.  
- **Reward shaping** → ricompense intermedie per prese utili e penalità per errori.  
- **Training loop** → simulazione di migliaia di partite per ottimizzare la policy.  

---

## 📂 Struttura del progetto

```bash
Tresette_AI/
│── .venv/              # Ambiente virtuale (non incluso nel repo)
│
│── obs/                # Moduli per osservazioni e rappresentazioni dello stato
│   └── encoder.py      # Encoder per trasformare lo stato in input per l'AI
│
│── tests/              # Test automatici e unit test
│
│── cards.py            # Rappresentazione e utilità per le carte
│── game4p.py           # Gestione del gioco a 4 giocatori
│── menu_cli.py         # Interfaccia a riga di comando per giocare
│── rules.py            # Regole del Tresette e funzioni di punteggio
│── train_dqn.py        # Script di training con Deep Q-Learning
│
└── README.md           # Documentazione del progetto
