# 🃏 Tresette AI

La nostra attività progettuale per il corso di Fondamenti di Intelligenza Artificiale si è basato sulla realizzazione di un sistema per la progettazione, l’allenamento e la valutazione di un’Intelligenza Artificiale capace di sfidare gli utenti al gioco di carte **Tresette**, tutto ciò applicando tecniche di **Reinforcement Learning (RL)** e simulazioni di partite.

---

## 🎯 Obiettivi

- **Modellare** le regole e le dinamiche del gioco del Tresette. -> Essere in grado di giocare contro un'AI non allenata(random)
- **Definire** un ambiente simulato in cui un agente AI possa giocare e migliorare.  -> Effettuare un training
- **Addestrare** l’agente tramite algoritmi di *Deep Reinforcement Learning* (es. Deep Q-Learning).
- **Integrare** una componente **euristica** per migliorare l’apprendimento. 
- **Permettere** a un giocatore umano di sfidare l’AI.  
- **Confrontare** strategie casuali, euristiche e di apprendimento.  

---

## 🃏 Regole implementate

- Gestione del **mazzo** e distribuzione delle carte.  
- Turni di gioco per **4 giocatori** attraverso lista di mosse legali.  
- Calcolo delle **prese** e dei **punteggi** secondo le regole del Tresette. -> 1/3 per le figure , 1 per l'asso
- Definizione delle **condizioni di vittoria**.  

---

## 🧠 Architettura AI

- **Ambiente** → rappresenta lo stato della partita (mani, prese, turno, carte giocate).  
- **Agente** → apprende a selezionare le mosse tramite `ε-greedy policy`.  
- **Reward shaping** → ricompense intermedie per prese utili e penalità per errori.  
- **Training loop** → simulazione di migliaia di partite per ottimizzare la policy.  

---
## 🏋️‍♂️ Training

Il processo di training dell’agente segue due fasi principali:  

1. **Fase casuale** → l’agente gioca utilizzando mosse casuali, in modo da esplorare lo spazio delle possibilità e raccogliere esperienza.  
2. **Fase euristica** → successivamente, viene introdotta un euristica che guida le scelte dell’agente (es. preferire mosse con carte forti o evitare sprechi), accelerando l’apprendimento prima che intervenga l’ottimizzazione tramite **Deep Q-Learning**.  

Durante il training:  
- Vengono salvati **checkpoint periodici** del modello, per poter riprendere l’allenamento senza perdere i progressi.  
- Se disponibile, viene utilizzata la **GPU** tramite **PyTorch** per velocizzare il processo di apprendimento.  

 I parametri di training (es. numero di episodi, learning rate, epsilon decay, frequenza dei checkpoint) possono essere modificati direttamente nel file train_dqn.py.

Per avviare il training:  
```bash
python train_dqn.py
```
---

## 🧩 Architettura generale

Il sistema è strutturato come un classico ambiente di **Reinforcement Learning**:  

| Componente | Descrizione |
|-------------|-------------|
| 🧠 Agente | Decide le mosse usando una `ε-greedy policy`. |
| 🎮 Ambiente | Simula lo stato della partita (mani, prese, turno, carte giocate). |
| 🪙 Reward shaping | Ricompense intermedie per prese utili e penalità per errori. |
| 🧩 Rete neurale (DQN) | Stima i valori Q e apprende la policy ottimale. |
| 🔁 Replay Buffer | Memorizza esperienze passate per stabilizzare l’apprendimento. |


---

## 📂 Struttura del progetto

```bash
Tresette_AI/
├─ cards.py                  # Carte e utilità
├─ rules.py                  # Regole e punteggi
├─ game4p.py                 # Logica del gioco a 4
├─ obs/encoder.py            # Codifica dello stato per la rete
├─ utils/HeuristicAgent.py   # Agente euristico
├─ train_dqn.py              # Training DQN + checkpoint
├─ watch_game.py             # Demo e tornei semplici
├─ watch_game_parallel.py    # Tornei batched (veloci su GPU)
├─ menu_cli.py               # Interfaccia testuale per giocare
└─ README.md
```
---

## 🚀 Installazione

Per installare ed eseguire il progetto:

```bash
# 1) Clona la repo
git clone https://github.com/BlackRaffo70/Tresette_AI.git
cd Tresette_AI

# 2) Crea e attiva un ambiente virtuale
python -m venv venv
source venv/bin/activate        # Mac/Linux
# .\venv\Scripts\activate       # Windows

# 3) Installa le dipendenze
pip install -r requirements.txt

# 4) (opzionale) Verifica GPU in PyTorch
python - << 'EOF'
import torch
print("CUDA disponibile:", torch.cuda.is_available())
EOF
```
---
## 🎮 Giocare contro l’AI

Dopo aver completato il training, è possibile sfidare l’agente tramite l’interfaccia a riga di comando:

```bash
python menu_cli.py
```

Oppure osservare partite simulate:

```bash
python Watch_game.py
```
Valutazione più rapida (batch, ideale su GPU)

```bash
python Watch_game_parallel.py
```


## 📦 Requisiti

Il progetto richiede **Python 3.10+** e le seguenti librerie principali:

```txt
Python >= 3.10
torch >= 2.0.0
numpy >= 1.23.0
matplotlib >= 3.7.0
tqdm >= 4.65.0
gym >= 0.26.0
```
---

# 👥 Autori

| | |
|:--:|:--:|
| <a href="https://github.com/BlackRaffo70"><img src="https://github.com/BlackRaffo70.png" width="110" alt="avatar Raffaele Neri"></a> | <a href="https://github.com/sebastianogiannitti"><img src="https://github.com/sebastianogiannitti" width="110" alt="avatar Sebastiano Giannitti"></a> |
| **Raffaele Neri**<br/>[@BlackRaffo70](https://github.com/BlackRaffo70) | **Sebastiano Giannitti**<br/>[@sebastianogiannitti](https://github.com/sebastianogiannitti) |

