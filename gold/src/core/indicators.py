"""
Moteur de calcul des indicateurs techniques (anciennement `dataAnalyser.py`).

PHILOSOPHIE
-----------
La classe `Indicators` est un objet *incrémental* et *streaming* : on lui pousse
une bougie à la fois via `push()` et elle met à jour tous ses buffers.
C'est indispensable pour deux raisons :

1. En production, le bot reçoit les prix un par un — on ne peut pas
   recalculer 10 ans d'historique à chaque tick.
2. À l'entraînement, cela garantit que chaque indicateur à l'instant t
   n'utilise QUE des données <= t. C'est la protection n°1 contre le
   « look-ahead bias », l'erreur qui fait qu'un modèle affiche 99 % de
   précision en backtest et perd de l'argent en réel.

CONVENTIONS
-----------
- Tous les buffers sont des `list[Dot]`, dans l'ordre chronologique croissant.
- Un buffer vide = « pas encore assez de données ». Les getters renvoient
  `None` dans ce cas, JAMAIS une valeur inventée (pas de zéro-padding : un
  zéro serait interprété par le modèle comme une vraie information).
- Toutes les périodes sont exprimées en NOMBRE DE BOUGIES, pas en jours.
  Sur des bougies H1 et un marché US ouvert ~6,5 h/jour, « SMA 20 jours »
  = 20 * 7 = 140 bougies environ. Voir `config/settings.py`.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from .entities import Dot, Quote


class Indicators:
    """Calcule et stocke les indicateurs techniques d'un symbole."""

    def __init__(self, symbol: str, max_history: int = 100_000) -> None:
        self.symbol = symbol

        # ------------------------------------------------------------------
        # Historique brut des bougies.
        # `deque(maxlen=...)` : borne mémoire. Sans cela, un bot qui tourne
        # des mois finit par saturer la RAM. Les indicateurs n'ont jamais
        # besoin de plus de quelques centaines de bougies en arrière.
        # ------------------------------------------------------------------
        self.quotes: Deque[Quote] = deque(maxlen=max_history)

        # ------------------------------------------------------------------
        # Buffers de sortie des indicateurs de tendance
        # ------------------------------------------------------------------
        self.sma: List[Dot] = []          # Moyenne mobile simple du close
        self.ema: List[Dot] = []          # Moyenne mobile exponentielle du close

        # ------------------------------------------------------------------
        # Oscillateurs (indicateurs bornés 0-100)
        # ------------------------------------------------------------------
        self.stoch_k: List[Dot] = []      # %K : oscillateur stochastique brut
        self.stoch_d: List[Dot] = []      # %D : SMA de %K (ligne de signal)
        self.rsi: List[Dot] = []          # Relative Strength Index

        # ------------------------------------------------------------------
        # Composants intermédiaires de l'ADX (Average Directional Index).
        # L'ADX est un indicateur en cascade : TR/DM -> lissage -> DI -> DX -> ADX.
        # On garde chaque étage séparé pour pouvoir déboguer/tracer.
        # ------------------------------------------------------------------
        self.true_range: List[Dot] = []
        self.dm_plus: List[Dot] = []      # Directional Movement haussier
        self.dm_minus: List[Dot] = []     # Directional Movement baissier
        self._tr_smoothed: List[Dot] = []
        self._dm_plus_smoothed: List[Dot] = []
        self._dm_minus_smoothed: List[Dot] = []
        self.dx: List[Dot] = []           # Directional Index
        self.adx: List[Dot] = []          # ADX = moyenne lissée du DX

    # ======================================================================
    # ACCESSEURS
    # ======================================================================
    @staticmethod
    def _last(buffer: List[Dot]) -> Optional[Dot]:
        """Dernière valeur d'un buffer, ou None s'il est vide.

        ⚠️ Bug corrigé : l'ancien code faisait `buffer[k]` après avoir vérifié
        seulement `len(buffer) == 0`. Avec `k = len - 1` calculé sur un AUTRE
        buffer, on pouvait lire un indice hors bornes ou, pire, une valeur
        désalignée dans le temps (donc une donnée du futur ou du passé lointain).
        """
        return buffer[-1] if buffer else None

    def last_quote(self) -> Optional[Quote]:
        return self.quotes[-1] if self.quotes else None

    def snapshot(self) -> Optional[dict]:
        """Renvoie la photo de TOUS les indicateurs à l'instant courant,
        ou None si au moins un indicateur n'est pas encore disponible.

        C'est la brique utilisée pour construire le dataset : soit la ligne
        est complète, soit on la jette. Jamais de valeur manquante remplie
        arbitrairement."""
        parts = {
            "quote": self.last_quote(),
            "sma": self._last(self.sma),
            "ema": self._last(self.ema),
            "rsi": self._last(self.rsi),
            "stoch_k": self._last(self.stoch_k),
            "stoch_d": self._last(self.stoch_d),
            "adx": self._last(self.adx),
        }
        if any(v is None for v in parts.values()):
            return None
        return parts

    # ======================================================================
    # POINT D'ENTRÉE PRINCIPAL
    # ======================================================================
    def push(self, quote: Quote, *, sma_period: int, stoch_k_period: int,
             stoch_d_period: int, rsi_period: int, adx_period: int) -> None:
        """Ajoute une bougie et met à jour tous les indicateurs.

        L'ordre des appels est important : l'ADX consomme la bougie
        précédente, le %D consomme le %K du même tick, etc.
        """
        self.quotes.append(quote)

        self._update_sma_ema(sma_period)
        self._update_stochastic(stoch_k_period)
        self._update_stoch_d(stoch_d_period)
        self._update_rsi(rsi_period)
        self._update_adx(adx_period)

    # ======================================================================
    # MOYENNES MOBILES
    # ======================================================================
    def _update_sma_ema(self, period: int) -> None:
        """SMA = moyenne arithmétique des `period` derniers close.
        EMA = moyenne pondérée exponentiellement, initialisée par la SMA.

        Formule EMA :  EMA_t = alpha * close_t + (1 - alpha) * EMA_{t-1}
        avec alpha = 2 / (period + 1)  (convention standard des plateformes).

        ⚠️ Bug corrigé : l'ancien `calculate_EMA` recalculait `alpha` à partir
        de `MIN_POINTS_SMA` (constante figée à 20) même quand on lui passait
        une période de 120. L'EMA « 120 » réagissait donc comme une EMA 20.
        Ici alpha est TOUJOURS dérivé de la période réellement demandée.
        """
        if len(self.quotes) < period:
            return  # pas assez d'historique : on ne publie rien

        window = list(self.quotes)[-period:]
        closes = [q.close_price for q in window]
        now = window[-1].time

        sma_value = float(np.mean(closes))
        self.sma.append(Dot(now, sma_value))

        alpha = 2.0 / (period + 1.0)
        if not self.ema:
            # Amorçage : la première EMA vaut la première SMA disponible.
            self.ema.append(Dot(now, sma_value))
        else:
            previous = self.ema[-1].value
            value = alpha * closes[-1] + (1.0 - alpha) * previous
            self.ema.append(Dot(now, value))

    @staticmethod
    def _rolling_sma(source: List[Dot], period: int, out: List[Dot]) -> None:
        """SMA générique appliquée à n'importe quelle série de Dot.
        Sert pour le %D (SMA du %K) et pour les lissages internes de l'ADX."""
        if len(source) < period:
            return
        window = source[-period:]
        out.append(Dot(window[-1].time, float(np.mean([d.value for d in window]))))

    @staticmethod
    def _wilder_smoothing(source: List[Dot], period: int, out: List[Dot]) -> None:
        """Lissage de Wilder — c'est le lissage utilisé par le VRAI ADX.

        Forme moyennée :  v_t = ( v_{t-1} * (period - 1) + brut_t ) / period
        Cela équivaut à une EMA d'alpha = 1/period (et non 2/(period+1)).
        On l'utilise pour le TR, les DM et l'ADX : comme les DI sont des
        RATIOS DM/TR, le facteur d'échelle se simplifie et le résultat est
        identique à la formule sommée de Wilder.

        ⚠️ Correction importante : l'ancien code lissait tout avec une EMA
        classique alpha = 2/(period+1), soit un lissage ~2x plus rapide.
        L'ADX obtenu était bien plus nerveux que celui de TradingView /
        MetaTrader, donc les seuils usuels (ADX > 25 = tendance établie)
        ne voulaient plus rien dire.
        """
        if len(source) < period:
            return
        if not out:
            # Amorçage : moyenne simple des `period` premières valeurs dispo.
            seed = float(np.mean([d.value for d in source[-period:]]))
            out.append(Dot(source[-1].time, seed))
            return
        previous = out[-1].value
        value = (previous * (period - 1) + source[-1].value) / period
        out.append(Dot(source[-1].time, value))

    # ======================================================================
    # OSCILLATEUR STOCHASTIQUE
    # ======================================================================
    def _update_stochastic(self, period: int) -> None:
        """%K = 100 * (close - plus_bas) / (plus_haut - plus_bas)

        Mesure où se situe le prix de clôture DANS le range récent :
        proche de 100 = clôture en haut du range (momentum haussier),
        proche de 0   = clôture en bas du range.

        ⚠️ Deux bugs corrigés :
        1. L'ancien `getMinMaxPrice` calculait min/max sur les CLOSE. Le
           stochastique standard utilise les plus HAUTS et plus BAS de la
           période (high/low), sinon le range est artificiellement étroit.
        2. La division par zéro (marché parfaitement plat) était attrapée
           par un `except:` nu qui masquait aussi les vraies erreurs. Ici on
           teste explicitement le cas et on publie 50 (milieu de range),
           qui est la convention.
        """
        if len(self.quotes) < period:
            return
        window = list(self.quotes)[-period:]
        lowest = min(q.low_price for q in window)
        highest = max(q.high_price for q in window)
        close = window[-1].close_price

        span = highest - lowest
        k = 50.0 if span == 0 else 100.0 * (close - lowest) / span
        self.stoch_k.append(Dot(window[-1].time, k))

    def _update_stoch_d(self, period: int) -> None:
        """%D = moyenne mobile simple du %K. C'est la ligne de signal :
        un croisement %K / %D est le déclencheur classique."""
        self._rolling_sma(self.stoch_k, period, self.stoch_d)

    # ======================================================================
    # RSI
    # ======================================================================
    def _update_rsi(self, period: int) -> None:
        """RSI = 100 - 100 / (1 + RS), avec RS = gain moyen / perte moyenne.

        Interprétation : > 70 = suracheté, < 30 = survendu.

        ⚠️ Bug corrigé : l'ancien slice `self.quoteBuffer[-period-1:-1]`
        EXCLUAIT la bougie la plus récente. Le RSI publié à l'instant t était
        donc en réalité le RSI de t-1 — un décalage d'une bougie, invisible
        à l'œil nu mais qui pourrit l'apprentissage du modèle.
        """
        if len(self.quotes) < period + 1:
            return
        closes = np.array([q.close_price for q in list(self.quotes)[-(period + 1):]],
                          dtype=np.float64)
        deltas = np.diff(closes)                     # variations bougie à bougie
        gains = deltas[deltas > 0].sum() / period
        losses = -deltas[deltas < 0].sum() / period

        if losses == 0:
            rsi = 100.0                              # que des hausses
        else:
            rs = gains / losses
            rsi = 100.0 - 100.0 / (1.0 + rs)
        self.rsi.append(Dot(self.quotes[-1].time, rsi))

    # ======================================================================
    # ADX (force de la tendance, indépendamment de son sens)
    # ======================================================================
    def _update_true_range(self) -> None:
        """True Range = max( H-L , |H - close_prec| , |L - close_prec| ).
        Prend en compte les « gaps » d'ouverture, contrairement à H-L seul."""
        current, previous = self.quotes[-1], self.quotes[-2]
        tr = max(
            current.high_price - current.low_price,
            abs(current.high_price - previous.close_price),
            abs(current.low_price - previous.close_price),
        )
        self.true_range.append(Dot(current.time, tr))

    def _update_directional_movement(self) -> None:
        """DM+ / DM- : quel côté du marché a le plus « poussé » cette bougie.

        Règle de Wilder : un seul des deux peut être non nul à la fois.
        """
        current, previous = self.quotes[-1], self.quotes[-2]
        up_move = current.high_price - previous.high_price
        down_move = previous.low_price - current.low_price

        dm_plus = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus = down_move if (down_move > up_move and down_move > 0) else 0.0

        self.dm_plus.append(Dot(current.time, dm_plus))
        self.dm_minus.append(Dot(current.time, dm_minus))

    def _update_adx(self, period: int) -> None:
        """Chaîne complète : TR/DM -> lissage Wilder -> DI± -> DX -> ADX."""
        if len(self.quotes) < 2:
            return
        self._update_true_range()
        self._update_directional_movement()

        self._wilder_smoothing(self.true_range, period, self._tr_smoothed)
        self._wilder_smoothing(self.dm_plus, period, self._dm_plus_smoothed)
        self._wilder_smoothing(self.dm_minus, period, self._dm_minus_smoothed)

        if not (self._tr_smoothed and self._dm_plus_smoothed and self._dm_minus_smoothed):
            return

        atr = self._tr_smoothed[-1].value
        if atr == 0:
            return  # marché figé : DI non défini

        di_plus = 100.0 * self._dm_plus_smoothed[-1].value / atr
        di_minus = 100.0 * self._dm_minus_smoothed[-1].value / atr

        denominator = di_plus + di_minus
        if denominator == 0:
            return
        dx = 100.0 * abs(di_plus - di_minus) / denominator
        self.dx.append(Dot(self.quotes[-1].time, dx))

        # L'ADX est la moyenne lissée du DX.
        self._wilder_smoothing(self.dx, period, self.adx)
