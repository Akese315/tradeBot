"""
Construction du jeu de features (anciennement éclaté entre `main.createDataset`
et `dataManager.analyse`, avec deux définitions incohérentes).

LE POINT LE PLUS IMPORTANT DE TOUT LE PROJET
--------------------------------------------
L'ancienne version donnait au LSTM des PRIX ABSOLUS en entrée
(`[open, high, low, close, sma, ema, os, osSMA, volume]`) et lui demandait de
prédire des PRIX ABSOLUS en sortie. C'est le piège classique du deep learning
appliqué au trading, pour trois raisons :

1. **Non-stationnarité.** NVDA valait 5 $ en 2015 et 900 $ en 2024. Le modèle
   entraîné sur la plage [5, 200] n'a jamais vu 900 et ne peut pas
   l'extrapoler : un réseau ne sait pas sortir de la distribution vue.

2. **Métrique trompeuse.** Sur des prix absolus, la solution optimale au sens
   MSE est « prédire le prix d'hier ». Le modèle atteint une loss minuscule
   et produit une courbe qui SEMBLE parfaitement coller à la réalité — mais
   décalée d'une bougie. C'est le fameux graphe « mon LSTM prédit la bourse »
   qui, dans les faits, ne prédit rien.

3. **Échelles hétérogènes.** Le volume est de l'ordre de 10^7, le RSI de 10^1.
   Sans normalisation, le gradient est écrasé par la feature la plus grande.

CORRECTION APPLIQUÉE : on travaille en **rendements logarithmiques** et en
grandeurs relatives, puis on standardise. La cible devient le rendement de la
bougie suivante — une variable approximativement stationnaire, centrée sur 0.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from config.settings import IndicatorConfig
from src.core.entities import Quote
from src.core.indicators import Indicators

# Ordre des colonnes — à conserver strictement identique entre
# l'entraînement et l'inférence, sinon le modèle lit du bruit.
FEATURE_COLUMNS: List[str] = [
    "log_return",        # rendement de la bougie : ln(close_t / close_{t-1})
    "high_over_close",   # amplitude haussière relative
    "low_over_close",    # amplitude baissière relative
    "close_over_sma",    # position du prix vs sa moyenne (>1 = au-dessus)
    "close_over_ema",
    "sma_slope",         # pente de la SMA = direction de la tendance
    "rsi",
    "stoch_k",
    "stoch_d",
    "adx",
    "log_volume_ratio",  # volume vs volume moyen récent
    "realized_vol",      # volatilité réalisée récente = identification du régime
]

# Colonne technique : sert à normaliser la cible bougie par bougie.
# Elle est aussi dans FEATURE_COLUMNS, donc le modèle la voit — c'est voulu :
# il doit savoir dans quel régime de volatilité il se trouve.
VOLATILITY_COLUMN = "realized_vol"

# Plancher de volatilité. Sans lui, une période anormalement calme
# (volatilité proche de 0) ferait exploser la cible normalisée : diviser par
# 0.00001 transforme un mouvement insignifiant en valeur à 3 chiffres, et le
# gradient part avec.
MIN_VOLATILITY = 1e-4

TARGET_COLUMN = "next_log_return"


def build_feature_frame(quotes: List[Quote],
                        indicator_config: IndicatorConfig | None = None,
                        verbose: bool = True) -> pd.DataFrame:
    """Transforme une liste de bougies en DataFrame de features + cible.

    Le parcours est strictement chronologique et incrémental : à l'indice k,
    seuls les indices <= k ont été vus. La seule exception est la CIBLE, qui
    regarde k+1 — c'est normal et c'est exactement ce qu'on veut apprendre.
    """
    config = indicator_config or IndicatorConfig()
    indicators = Indicators(symbol="dataset")
    records: List[dict] = []

    total = len(quotes)
    for k in range(total - 1):          # -1 : la dernière bougie n'a pas de cible
        quote = quotes[k]
        next_quote = quotes[k + 1]

        indicators.push(
            quote,
            sma_period=config.sma_period,
            stoch_k_period=config.stoch_k_period,
            stoch_d_period=config.stoch_d_period,
            rsi_period=config.rsi_period,
            adx_period=config.adx_period,
        )

        snap = indicators.snapshot()
        if snap is None:
            # Période de « chauffe » : tant que le plus lent des indicateurs
            # n'a pas assez d'historique, la ligne est incomplète -> on saute.
            continue
        if k == 0:
            continue

        previous = quotes[k - 1]
        if previous.close_price <= 0 or quote.close_price <= 0:
            continue  # donnée corrompue (prix nul/négatif) : on écarte

        sma_value = snap["sma"].value
        ema_value = snap["ema"].value

        # Pente de la SMA sur les 2 derniers points (0 si indisponible).
        sma_slope = 0.0
        if len(indicators.sma) >= 2 and indicators.sma[-2].value != 0:
            sma_slope = (sma_value / indicators.sma[-2].value) - 1.0

        # Volume relatif : log(1+v) évite le log(0) sur les bougies sans échange.
        # --- Volatilité réalisée : écart-type des 20 derniers rendements ----
        # C'est la feature d'identification du RÉGIME de marché. Sans elle, le
        # modèle ne peut pas savoir s'il opère dans un marché calme ou agité,
        # et donc pas adapter son comportement.
        # Calculée UNIQUEMENT sur des bougies passées : aucune fuite.
        recent_closes = [q.close_price for q in list(indicators.quotes)[-21:]]
        if len(recent_closes) >= 21:
            recent_returns = np.diff(np.log(recent_closes))
            realized_vol = float(np.std(recent_returns))
        else:
            realized_vol = MIN_VOLATILITY
        realized_vol = max(realized_vol, MIN_VOLATILITY)

        recent_volumes = [q.volume for q in list(indicators.quotes)[-20:]]
        mean_volume = float(np.mean(recent_volumes)) if recent_volumes else 0.0
        log_volume_ratio = (
            np.log1p(quote.volume) - np.log1p(mean_volume) if mean_volume > 0 else 0.0
        )

        records.append({
            "time": quote.time,
            "close": quote.close_price,      # conservé pour reconstruire le prix
            "log_return": np.log(quote.close_price / previous.close_price),
            "high_over_close": quote.high_price / quote.close_price - 1.0,
            "low_over_close": quote.low_price / quote.close_price - 1.0,
            "close_over_sma": quote.close_price / sma_value - 1.0 if sma_value else 0.0,
            "close_over_ema": quote.close_price / ema_value - 1.0 if ema_value else 0.0,
            "sma_slope": sma_slope,
            "rsi": snap["rsi"].value / 100.0,        # ramené dans [0, 1]
            "stoch_k": snap["stoch_k"].value / 100.0,
            "stoch_d": snap["stoch_d"].value / 100.0,
            "adx": snap["adx"].value / 100.0,
            "log_volume_ratio": float(log_volume_ratio),
            "realized_vol": realized_vol,
            # CIBLE : rendement de la bougie SUIVANTE.
            TARGET_COLUMN: np.log(next_quote.close_price / quote.close_price),
        })

        if verbose and k % 500 == 0:
            print(f"\tConstruction du dataset : {100 * k // total} %", end="\r")

    frame = pd.DataFrame.from_records(records)
    if verbose:
        print(f"\tConstruction du dataset : 100 % — {len(frame)} lignes exploitables")

    # Filet de sécurité : un inf ou un NaN qui passe jusqu'au modèle produit
    # une loss = NaN et détruit tous les poids en une seule rétropropagation.
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame


def rows_to_quotes(rows: List[Tuple]) -> List[Quote]:
    """Convertit les lignes SQL (time, open, high, low, close, volume) en Quote.

    ⚠️ Bug corrigé : l'ancien `arrayToQuote` faisait
    `array[i][0].strptime("%Y-%m-%d %H:%M:%S")`.
    `strptime` est une méthode de CLASSE qui attend (chaîne, format) ;
    l'appeler sur un objet datetime avec un seul argument lève une TypeError.
    Le driver MySQL renvoie déjà des `datetime` : aucune conversion nécessaire.
    """
    return [
        Quote(
            time=row[0],
            open_price=float(row[1]),
            high_price=float(row[2]),
            low_price=float(row[3]),
            close_price=float(row[4]),
            volume=float(row[5] or 0.0),
        )
        for row in rows
    ]