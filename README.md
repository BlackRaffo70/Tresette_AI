<img width="896" height="208" alt="image" src="https://github.com/user-attachments/assets/8970c4f2-33d3-4192-9a33-d0c21f1ace12" />




La nostra attività progettuale per il corso di Fondamenti di Intelligenza Artificiale si è basato sulla realizzazione di un sistema per la progettazione, l’allenamento e la valutazione di un’Intelligenza Artificiale capace di sfidare gli utenti al gioco di carte **Tresette**, tutto ciò applicando tecniche di **Reinforcement Learning (RL)** e simulazioni di partite.

---

## 🎯 Obiettivi

- **Modellare** le regole e le dinamiche del gioco del Tresette. -> Essere in grado di giocare contro un'AI non allenata(random)
- **Definire** un ambiente simulato in cui un agente AI possa giocare e migliorare.  -> Effettuare un training
- **Addestrare** l’agente tramite algoritmi di *Deep Reinforcement Learning* (es. Deep Q-Learning), attraverso simulazioni di partite ed esplorazione contro AI che inizialmente gioca in random, poi utilizza un heuristica e in fine con una fase DQN pura
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

- **Ambiente** → rappresenta lo stato completo della partita (mani, prese, turno, carte giocate e segnali).  
- **Agente** → apprende a selezionare la mossa ottimale attraverso una `ε-greedy policy`, bilanciando esplorazione e sfruttamento.  
- **Rete DQN** → stima i valori Q delle azioni possibili e aggiorna i pesi della rete tramite backpropagation.  
- **Reward shaping** → fornisce ricompense dense per incoraggiare prese vantaggiose e penalizzare mosse deboli o errori strategici.  
- **Training loop** → simula migliaia di partite, aggiornando la policy e salvando checkpoint periodici per monitorare l’evoluzione dell’agente.   

---
## 🏋️‍♂️ Training

Il processo di training dell’agente segue 3 fasi principali:  

Il processo di training dell’agente segue 3 fasi principali:  

1. **Fase casuale (warm-up iniziale)** → l’agente gioca utilizzando mosse casuali o parzialmente guidate da regole semplici, così da esplorare lo spazio delle possibilità e riempire il replay buffer con le prime esperienze.  

2. **Fase euristica (pre-training)** → in questa fase, l’agente segue esclusivamente la logica dell’euristica, che privilegia mosse più sensate (es. conservare gli assi, evitare di sprecare carte forti). Il DQN osserva queste partite e impara da esempi coerenti.  

3. **Fase DQN pura (sfruttamento)** → una volta terminato il pre-training, l’agente utilizza solo la rete neurale per scegliere le mosse, basandosi sui valori Q stimati. In questa fase non avviene più esplorazione casuale: l’AI gioca in modo deterministico, sfruttando al massimo la policy appresa.  

Durante il training:  
- Possono essere salvati **checkpoint periodici** del modello, per poter riprendere l’allenamento senza perdere i progressi.  
- Se disponibile, viene utilizzata la **GPU** tramite **PyTorch** per velocizzare il processo di apprendimento.  

 I parametri di training (es. numero di episodi, learning rate, epsilon decay, frequenza dei checkpoint) possono essere modificati direttamente nel file train_dqn.py.
 
 Abbiamo sfruttato le **infrastrutture HPC fornite da Università di Bologna (CS UNIBO)** per il training dell’agente, in particolare utilizzando una GPU NVIDIA L40 presente nella partizione “l40”. Questa configurazione ha permesso di accelerare significativamente l’addestramento del modello DQN garantendo tempi di calcolo adeguati e sfruttando al meglio il batch-processing parallelo.

Per avviare il training:  
```bash
python train_dqn.py
```
---

## 🧩 Architettura generale

Il sistema è strutturato come un classico ambiente di **Reinforcement Learning**:  

| Componente | Descrizione |
|-------------|-------------|
| 🧠 Agente | Decide le mosse in base ai valori Q stimati dalla rete neurale. |
| 🎮 Ambiente | Simula lo stato del gioco, gestendo mani, prese, punteggi e regole del Tresette. |
| 🪙 Reward shaping | Introduce ricompense intermedie per incoraggiare comportamenti utili (es. vincere prese, evitare errori). |
| 🧩 Rete neurale (DQN) | Predice i Q-values per ogni azione possibile e aggiorna i pesi durante il training. |
| 🔁 Replay Buffer | Memorizza le esperienze (stato, azione, ricompensa, stato successivo) per stabilizzare l’apprendimento. |
| 🧮 Target Network | Copia periodicamente i pesi della rete principale per evitare oscillazioni e migliorare la convergenza. |
| ⚙️ Epsilon Decay | Riduce gradualmente la casualità nelle scelte, passando da esplorazione a sfruttamento. |
| 🧑‍🏫 Agente Euristico | Fornisce esempi iniziali sensati per accelerare l’apprendimento dell’agente DQN. |


---

## 📂 Struttura del progetto

```bash
Tresette_AI/
├─ cards.py                  # Definizione e utilità per le carte
├─ rules.py                  # Regole del Tresette e calcolo dei punteggi
├─ game4p.py                 # Logica del gioco a 4 giocatori e gestione dei turni
├─ obs/encoder.py            # Codifica dello stato di gioco in feature numeriche
├─ utils/HeuristicAgent.py   # Agente basato su regole euristiche
├─ train_dqn.py              # Training DQN, gestione checkpoint e salvataggi
├─ watch_game.py             # Simulazione e visualizzazione di partite singole
├─ watch_game_parallel.py    # Esecuzione di tornei paralleli (batch GPU)
├─ menu_cli.py               # Interfaccia a riga di comando per giocare contro l’AI
├─ train_long.sbatch         # Script per esecuzione su cluster HPC (GPU L40 nel nostro caso)
├─ requirements.txt          # Dipendenze del progetto
└─ README.md                 # Documentazione principale
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
| <a href="https://github.com/BlackRaffo70"><img src="https://github.com/BlackRaffo70.png" width="110" alt="avatar Raffaele Neri"></a> | <a href="https://github.com/sebastianogiannitti"><img src="https://github.com/sebastianogiannitti.png" width="110" alt="avatar Sebastiano Giannitti"></a> |
| **Raffaele Neri**<br/>[@BlackRaffo70](https://github.com/BlackRaffo70) | **Sebastiano Giannitti**<br/>[@sebastianogiannitti](https://github.com/sebastianogiannitti) |

