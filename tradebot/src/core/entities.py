"""
Structures de données de base du projet (anciennement `utility.py`).

Ces objets sont volontairement « bêtes » : ils ne contiennent aucune logique
métier, seulement des données. Toute la logique de calcul vit dans
`src/core/indicators.py`.

Pourquoi des dataclasses plutôt que des classes classiques ?
  - `__init__`, `__repr__` et `__eq__` sont générés automatiquement ;
  - `frozen=True` rend l'objet immuable : une bougie déjà passée ne doit
    JAMAIS être modifiée après coup (c'est une source classique de fuite de
    données / "look-ahead bias" en trading algorithmique).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Quote:
    """Une bougie OHLCV (Open, High, Low, Close, Volume) sur une période donnée.

    Attributs :
        open_price  : prix d'ouverture de la période
        high_price  : prix le plus haut de la période
        low_price   : prix le plus bas de la période
        close_price : prix de clôture de la période
        time        : horodatage de FIN de la période (important pour l'alignement)
        volume      : volume échangé sur la période

    ⚠️ Piège corrigé : dans l'ancien code, l'ordre des arguments du constructeur
    (`open, close, high, low, time, volume`) n'était pas le même partout — dans
    `bot.py` le volume n'était même pas passé, ce qui levait une TypeError.
    Ici les champs sont nommés, donc l'erreur devient impossible.
    """

    open_price: float
    high_price: float
    low_price: float
    close_price: float
    time: datetime
    volume: float

    @property
    def typical_price(self) -> float:
        """Prix « typique » (HLC/3), souvent utilisé à la place du close
        pour lisser les indicateurs (VWAP, CCI, etc.)."""
        return (self.high_price + self.low_price + self.close_price) / 3.0

    @property
    def true_range_intraday(self) -> float:
        """Amplitude de la bougie seule (High - Low).
        Ce n'est PAS le True Range complet : le vrai TR a besoin de la bougie
        précédente (voir `Indicators.calculate_true_range`)."""
        return self.high_price - self.low_price


@dataclass(frozen=True, slots=True)
class Dot:
    """Un point d'une série temporelle : (instant, valeur).

    Utilisé pour stocker la sortie de tous les indicateurs (SMA, EMA, RSI…).
    Garder le temps à côté de la valeur évite les décalages d'indices quand
    un indicateur démarre plus tard qu'un autre (ex : la SMA(120) n'a pas de
    valeur avant la 120e bougie)."""

    time: datetime
    value: float