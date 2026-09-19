"""
Analyse multi-timeframe.

PRINCIPE
--------
On prédit toujours sur UN timeframe (le « base », typiquement H1). Les autres
timeframes servent de CONTEXTE : à chaque bougie H1, on ajoute l'état du 4 h,
du journalier et de l'hebdomadaire.

Le modèle voit alors, pour une même bougie :
    « le prix est au-dessus de sa moyenne en H1, sous sa moyenne en journalier,
      le RSI hebdomadaire est en surachat, l'ADX 4 h monte »

C'est exactement le raisonnement d'un trader qui empile plusieurs écrans.

LE PIÈGE, ET COMMENT IL EST TRAITÉ
-----------------------------------
Une bougie est horodatée à son OUVERTURE (convention MetaTrader et de la
quasi-totalité des fournisseurs). La bougie journalière du 1er juillet porte
l'horodatage `2004-07-01 00:00` mais ne se termine qu'à la fin de la journée.

Utiliser son `close` à 10 h du matin, c'est lire un prix qui n'existe pas
encore. Le modèle apprendrait à « prédire » un futur qu'on lui a donné, ses
scores exploseraient, et tout s'effondrerait en réel.

Protection appliquée ici : chaque bougie de contexte reçoit un champ
`available_at = ouverture + durée`. L'alignement (`merge_asof` en mode
`backward`) ne peut donc rattacher à une bougie H1 que des bougies de contexte
DÉJÀ CLÔTURÉES. Une bougie journalière n'influence les H1 qu'à partir du
lendemain.

C'est conservateur, et c'est le seul réglage défendable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import IndicatorConfig
from src.core.entities import Quote
from src.core.indicators import Indicators

# Durée d'une bougie, en minutes. Les clés correspondent aux suffixes des
# fichiers (XAU_4h_data.csv -> "4h").
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
    "1Month": 43200,
}

# Périodes des indicateurs de contexte, en NOMBRE DE BOUGIES.
# Contrairement au timeframe de base, on n'essaie pas de convertir en jours :
# sur un graphique journalier, « SMA 20 » veut dire 20 bougies, point. C'est
# la convention de toutes les plateformes, et elle est plus lisible.
CONTEXT_INDICATORS = IndicatorConfig(
    sma_period=20,
    stoch_k_period=14,
    stoch_d_period=3,
    rsi_period=14,
    adx_period=14,
)

# Features extraites de chaque timeframe de contexte.
# Volontairement peu nombreuses : avec 3 timeframes de contexte, 5 features
# chacun font déjà 15 colonnes supplémentaires. Au-delà, on dilue le signal
# et on invite le sur-apprentissage.
CONTEXT_FEATURES = ["ret", "over_sma", "rsi", "adx", "vol"]


def build_context_features(quotes: List[Quote], timeframe: str,
                           prefix: Optional[str] = None) -> pd.DataFrame:
    """Calcule les features d'un timeframe de contexte.

    Renvoie un DataFrame avec `available_at` (instant à partir duquel la
    bougie est utilisable) et les colonnes préfixées.
    """
    prefix = prefix or timeframe
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Timeframe inconnu : {timeframe}. "
                         f"Connus : {list(TIMEFRAME_MINUTES)}")
    duration = pd.Timedelta(minutes=minutes)

    indicators = Indicators(symbol=f"ctx_{prefix}")
    records: List[dict] = []

    for index, quote in enumerate(quotes):
        indicators.push(
            quote,
            sma_period=CONTEXT_INDICATORS.sma_period,
            stoch_k_period=CONTEXT_INDICATORS.stoch_k_period,
            stoch_d_period=CONTEXT_INDICATORS.stoch_d_period,
            rsi_period=CONTEXT_INDICATORS.rsi_period,
            adx_period=CONTEXT_INDICATORS.adx_period,
        )
        snap = indicators.snapshot()
        if snap is None or index == 0:
            continue

        previous = quotes[index - 1]
        if previous.close_price <= 0 or quote.close_price <= 0:
            continue

        sma_value = snap["sma"].value
        closes = [q.close_price for q in list(indicators.quotes)[-21:]]
        if len(closes) >= 21:
            realized = float(np.std(np.diff(np.log(closes))))
        else:
            realized = 0.0

        records.append({
            # Clé d'alignement : PAS l'ouverture, mais la clôture.
            "available_at": quote.time + duration,
            # Heure d'ouverture réelle, conservée UNIQUEMENT pour le contrôle
            # anti-fuite. La dériver d'`available_at` rendrait le contrôle
            # circulaire : il ne pourrait pas détecter un décalage erroné.
            "bar_open": quote.time,
            f"{prefix}_ret": np.log(quote.close_price / previous.close_price),
            f"{prefix}_over_sma": (quote.close_price / sma_value - 1.0
                                   if sma_value else 0.0),
            f"{prefix}_rsi": snap["rsi"].value / 100.0,
            f"{prefix}_adx": snap["adx"].value / 100.0,
            f"{prefix}_vol": realized,
        })

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(
            f"Timeframe {timeframe} : aucune bougie exploitable. "
            f"Il en faut au moins {CONTEXT_INDICATORS.adx_period * 2} pour "
            "que l'ADX se stabilise.")
    return frame.sort_values("available_at").reset_index(drop=True)


def attach_context(base: pd.DataFrame,
                   contexts: Dict[str, pd.DataFrame],
                   verbose: bool = True) -> tuple[pd.DataFrame, List[str]]:
    """Rattache les features de contexte au DataFrame du timeframe de base.

    `base` doit contenir une colonne `time`.
    Renvoie (DataFrame enrichi, liste des colonnes de contexte ajoutées).
    """
    merged = base.sort_values("time").reset_index(drop=True)
    added: List[str] = []

    for name, context in contexts.items():
        # `bar_open` sert au contrôle, pas au modèle : on ne le fusionne pas.
        context = context.drop(columns=["bar_open"], errors="ignore")
        columns = [c for c in context.columns if c != "available_at"]

        # `direction="backward"` : on prend la dernière bougie de contexte
        # dont `available_at` est <= `time`. Comme `available_at` est déjà la
        # clôture, la bougie est nécessairement terminée. Aucune fuite possible.
        merged = pd.merge_asof(
            merged,
            context,
            left_on="time",
            right_on="available_at",
            direction="backward",
        )
        merged = merged.drop(columns=["available_at"])
        added.extend(columns)

        if verbose:
            covered = merged[columns[0]].notna().mean()
            print(f"  contexte {name:<8} {len(context):>7} bougies  "
                  f"couverture {covered:.1%}")

    before = len(merged)
    # Les premières bougies du base n'ont pas encore de contexte disponible
    # (le timeframe hebdomadaire met plusieurs mois à produire sa première
    # valeur d'ADX). On les écarte plutôt que de les remplir arbitrairement.
    merged = merged.dropna(subset=added).reset_index(drop=True)
    if verbose and before != len(merged):
        print(f"  {before - len(merged)} bougies écartées "
              "(contexte pas encore disponible)")

    return merged, added


def verify_no_lookahead(base_times: pd.Series, context: pd.DataFrame,
                        timeframe: str) -> None:
    """Contrôle explicite : aucune bougie de contexte n'est utilisée avant
    sa clôture réelle.

    ATTENTION AU PIÈGE MÉTHODOLOGIQUE. Une première version de ce contrôle
    comparait `available_at` à `time` — or `merge_asof(direction="backward")`
    garantit DÉJÀ que `available_at <= time`. Le test était tautologique :
    il réussissait toujours, y compris sur des données volontairement
    sabotées. Un contrôle qui ne peut pas échouer ne contrôle rien.

    La version correcte compare l'heure d'OUVERTURE réelle de la bougie
    (`bar_open`, conservée telle quelle depuis la source) à laquelle on
    rajoute la durée du timeframe. Cette valeur est indépendante de
    `available_at`, donc un décalage erroné devient détectable.
    """
    if "bar_open" not in context.columns:
        raise ValueError(
            "Contrôle impossible : colonne `bar_open` absente. "
            "Elle doit être produite par build_context_features().")

    duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])

    aligned = pd.merge_asof(
        pd.DataFrame({"time": base_times.sort_values().reset_index(drop=True)}),
        context[["available_at", "bar_open"]],
        left_on="time", right_on="available_at", direction="backward",
    ).dropna()

    # Instant réel de clôture, calculé depuis la source et non depuis la clé.
    real_close = aligned["bar_open"] + duration
    violations = int((real_close > aligned["time"]).sum())

    if violations:
        worst = (real_close - aligned["time"]).max()
        raise AssertionError(
            f"FUITE DÉTECTÉE sur {timeframe} : {violations} bougies de base "
            f"utilisent une bougie de contexte non clôturée "
            f"(avance maximale : {worst}). "
            "Vérifiez le décalage `available_at = ouverture + durée`.")
