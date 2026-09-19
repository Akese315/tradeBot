# TradeBot

Bot de trading algorithmique : collecte de données de marché, calcul
d'indicateurs techniques, entraînement d'un modèle LSTM (PyTorch) et
journalisation temps réel.

> **Avertissement.** Ce projet est un exercice d'ingénierie et de recherche.
> Il ne passe aucun ordre et ne constitue pas un conseil en investissement.
> Prédire les rendements à court terme d'actions liquides est l'un des
> problèmes les plus difficiles qui soient : la quasi-totalité des modèles
> qui « fonctionnent » en backtest contiennent une fuite de données.

---

## 1. Arborescence

```
tradebot/
├── main.py                  # CLI unique : harvest / delete / train / run
├── config/
│   └── settings.py          # secrets, chemins, périodes, hyperparamètres
├── sql/
│   └── schema.sql           # schéma de la base MySQL
├── src/
│   ├── core/                # domaine métier, sans dépendance externe
│   │   ├── entities.py      # Quote, Dot
│   │   └── indicators.py    # SMA, EMA, RSI, Stochastique, ADX
│   ├── data/                # entrées/sorties
│   │   ├── database.py      # accès MySQL (requêtes paramétrées)
│   │   └── providers.py     # Finnhub (live) + AlphaVantage (historique)
│   ├── ai/
│   │   ├── features.py      # bougies -> matrice de features
│   │   ├── dataset.py       # fenêtres glissantes, découpage, normalisation
│   │   ├── model.py         # architecture LSTM
│   │   └── train.py         # boucle d'entraînement + évaluation
│   └── bot.py               # orchestration temps réel
├── data/                    # (généré) raw/ et features/
└── models/                  # (généré) un dossier par entraînement
```

**Principe de dépendance** : `core` ne dépend de rien, `data` et `ai`
dépendent de `core`, `bot` et `main` dépendent de tout. Les flèches ne
pointent jamais dans l'autre sens — c'est ce qui rend le domaine testable
sans base de données ni réseau.

---

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # puis remplissez vos identifiants
mysql -u root -p < sql/schema.sql
```

---

## 3. Utilisation

```bash
# Collecte de l'historique horaire d'une année
python main.py harvest --symbol NVDA --year 2023

# Suppression (totale ou par année)
python main.py delete --symbol NVDA --year 2023

# Entraînement du modèle
python main.py train --symbol NVDA

# Bot temps réel (observation + journalisation, aucun ordre)
python main.py run --symbol NVDA --interval 60
```

---

## 4. Bugs corrigés lors de la réorganisation

| # | Fichier d'origine | Problème | Gravité |
|---|---|---|---|
| 1 | `bot.py` | Clé API et mot de passe MySQL en clair dans le code | Critique |
| 2 | `bot.py` | `cursor.close()` / `conn.close()` au niveau module : `main.py` héritait d'un curseur fermé | Bloquant |
| 3 | `AI.py` | `tradingModel()` n'existe pas (classe = `TradingModel`, 4 arguments requis) ; `model.py` jamais importé | Bloquant |
| 4 | `AI.py` | `inputs.unsqueeze(1)` -> séquences de longueur 1 : le LSTM n'avait aucune mémoire temporelle | Critique |
| 5 | `AI.py` / `main.py` | Prix absolus en entrée et en sortie, sans normalisation : cible non stationnaire, apprentissage impossible | Critique |
| 6 | `AI.py` | Aucune loss de validation, aucun early stopping, évaluation d'un `.pth` rechargé du disque | Majeur |
| 7 | `main.py` | `datetime.strptime()` appelé sur un objet `datetime` avec un seul argument | Bloquant |
| 8 | `main.py` | `dm.getLocalTrainingData(symbol)` appelé sans le paramètre `cursor` | Bloquant |
| 9 | `dataManager.py` | `cleanTrainingData(symbol, year)` appelé avec 2 arguments sur 4 | Bloquant |
| 10 | `dataManager.py` | SQL construit par concaténation de chaînes (injection) | Majeur |
| 11 | `dataManager.py` | `min(rows)` compare des tuples, pas des prix | Majeur |
| 12 | `dataManager.py` | `conn.commit()` à chaque ligne insérée | Performance |
| 13 | `dataAnalyser.py` | RSI calculé sur `[-period-1:-1]` : décalé d'une bougie | Majeur |
| 14 | `dataAnalyser.py` | Stochastique calculé sur les close au lieu des high/low | Majeur |
| 15 | `dataAnalyser.py` | EMA : `alpha` toujours dérivé de `MIN_POINTS_SMA=20`, quelle que soit la période demandée | Majeur |
| 16 | `dataAnalyser.py` | ADX lissé en EMA classique au lieu du lissage de Wilder | Moyen |
| 17 | `dataAnalyser.py` | `except:` nu masquant les vraies erreurs | Moyen |
| 18 | `bot.py` | `Quote(...)` construit sans le `volume` -> TypeError | Bloquant |
| 19 | `bot.py` | Finnhub `pc` (clôture de la veille) utilisé comme close courant | Majeur |
| 20 | `database.sql` | Aucune contrainte d'unicité : collecte relancée = historique dupliqué | Majeur |
| 21 | `model.py` + `AI.py` | `create_batches` dupliqué à l'identique | Maintenance |

---

## 5. Prochaines étapes recommandées

Dans cet ordre, sans sauter d'étape :

1. **Backtest avec frais et slippage.** Une précision directionnelle de 55 %
   ne rapporte rien si le spread mange le gain moyen par trade.
2. **Validation croisée temporelle glissante** (walk-forward), pas un seul
   découpage : un unique split peut être chanceux.
3. **Étude par régime.** Séparer les périodes de tendance (ADX > 25) et de
   range : un modèle unique dilue deux dynamiques opposées.
4. **Multi-symboles.** Entraîner sur un panier de titres corrélés multiplie
   les données et réduit fortement le sur-apprentissage.
5. **Ensuite seulement**, envisager le passage d'ordres — en papier trading
   pendant plusieurs mois d'abord.


On étudie sur le temps passé (CSV) :
python scripts/train_from_csv.py --symbol NVDA --csv-dir data/dataframe


Comparez l'historique des paramètres : 
grep "RUN :\|loss_test\|bat_la_baseline\|precision_dir\|meilleure_epoque" historique_runs.txt