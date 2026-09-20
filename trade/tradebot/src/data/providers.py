"""
Fournisseurs de données de marché.

Deux sources, deux rôles bien distincts :
  - **Finnhub**      : cotation TEMPS RÉEL (le bot en production).
  - **AlphaVantage** : historique intraday (la constitution du dataset).

Séparer les deux dans des classes distinctes permet d'en changer une sans
toucher au reste (principe d'inversion de dépendance : le bot dépend de
l'interface, pas du fournisseur).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Iterator, List, Optional, Tuple

import requests

from config.settings import ALPHAVANTAGE_KEY, FINNHUB_KEY
from src.core.entities import Quote


class MarketDataError(RuntimeError):
    """Erreur remontée quand une API refuse ou renvoie une réponse inattendue.

    ⚠️ L'ancien code se contentait de `print(response.status_code)` puis
    renvoyait `None` implicitement. Le `None` remontait jusqu'aux indicateurs
    et provoquait une AttributeError 40 lignes plus loin, impossible à
    diagnostiquer. Ici on échoue *fort et tôt*, avec un message explicite.
    """


# ==========================================================================
# TEMPS RÉEL — FINNHUB
# ==========================================================================
class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str = FINNHUB_KEY, timeout: int = 10) -> None:
        if not api_key:
            raise MarketDataError("FINNHUB_KEY absente du .env")
        self.timeout = timeout
        # `Session` réutilise la connexion TCP : ~3x plus rapide en boucle.
        self.session = requests.Session()
        self.session.headers.update({"X-Finnhub-Token": api_key})

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = self.session.get(
            f"{self.BASE_URL}{path}", params=params, timeout=self.timeout
        )
        if response.status_code != 200:
            raise MarketDataError(
                f"Finnhub {path} a répondu {response.status_code} : {response.text[:200]}"
            )
        return response.json()

    def get_quote(self, symbol: str) -> Quote:
        """Cotation instantanée.

        Correspondance des champs Finnhub :
            o = open du jour, h = high du jour, l = low du jour,
            c = prix courant, pc = clôture précédente, t = timestamp UNIX.

        ⚠️ Bug corrigé : l'ancien code mappait `pc` (clôture de la VEILLE) sur
        le champ `closePrice` de la bougie courante. Tous les indicateurs
        travaillaient donc sur le prix d'hier, décalé d'un jour. On utilise
        maintenant `c`, le prix courant, qui est le bon équivalent du close
        pour une bougie en cours de formation.
        """
        payload = self._get("/quote", {"symbol": symbol})
        return Quote(
            open_price=float(payload["o"]),
            high_price=float(payload["h"]),
            low_price=float(payload["l"]),
            close_price=float(payload["c"]),
            time=datetime.fromtimestamp(payload["t"]),
            volume=float(payload.get("v", 0.0)),  # Finnhub /quote ne renvoie pas
                                                  # toujours le volume : défaut 0.
        )

    def is_market_open(self, exchange: str = "US") -> bool:
        return bool(self._get("/stock/market-status", {"exchange": exchange})["isOpen"])

    def symbol_exists(self, symbol: str) -> bool:
        return self._get("/search", {"q": symbol}).get("count", 0) > 0


# ==========================================================================
# HISTORIQUE — ALPHAVANTAGE
# ==========================================================================
class AlphaVantageClient:
    BASE_URL = "https://www.alphavantage.co/query"

    # Plan gratuit : 5 requêtes/minute. 15 s d'attente = marge de sécurité.
    THROTTLE_SECONDS = 15

    def __init__(self, api_key: str = ALPHAVANTAGE_KEY, timeout: int = 30) -> None:
        if not api_key:
            raise MarketDataError("ALPHAVANTAGE_KEY absente du .env")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_month(self, symbol: str, month: str,
                    interval: str = "60min") -> List[Tuple]:
        """Télécharge un mois de bougies intraday.

        `month` au format 'AAAA-MM'. Renvoie des tuples prêts pour
        `insert_training_rows` : (time, symbol, open, high, low, close, volume).
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "month": month,
            "apikey": self.api_key,
            "outputsize": "full",
        }
        payload = self.session.get(self.BASE_URL, params=params,
                                   timeout=self.timeout).json()

        # AlphaVantage renvoie un HTTP 200 même en cas d'erreur : le message
        # est caché dans une clé "Note" (quota) ou "Error Message" (symbole
        # inconnu). Il FAUT tester ces clés avant de parser.
        if "Note" in payload:
            raise MarketDataError(f"Quota AlphaVantage atteint : {payload['Note']}")
        if "Error Message" in payload:
            raise MarketDataError(f"Requête refusée : {payload['Error Message']}")

        series_key = f"Time Series ({interval})"
        if series_key not in payload:
            raise MarketDataError(f"Réponse inattendue, clés = {list(payload)}")

        rows: List[Tuple] = []
        # `sorted()` : AlphaVantage renvoie du plus récent au plus ancien.
        # L'ancien code faisait `items[::-1]`, ce qui suppose que le dict est
        # déjà trié — vrai en pratique, mais fragile. Un tri explicite est sûr.
        for timestamp, values in sorted(payload[series_key].items()):
            rows.append((
                timestamp,
                symbol,
                float(values["1. open"]),
                float(values["2. high"]),
                float(values["3. low"]),
                float(values["4. close"]),
                float(values["5. volume"]),
            ))
        return rows

    def iter_year(self, symbol: str, year: int,
                  interval: str = "60min") -> Iterator[Tuple[str, List[Tuple]]]:
        """Générateur : produit (mois, lignes) pour toute une année.

        Un générateur plutôt qu'une liste : on insère en base au fil de l'eau
        au lieu de garder 12 mois de données en mémoire, et une interruption
        ne fait pas perdre le travail déjà fait.
        """
        today = datetime.now()
        last_month = today.month if year == today.year else 12
        for month_index in range(1, last_month + 1):
            month = f"{year}-{month_index:02d}"
            yield month, self.fetch_month(symbol, month, interval)
            time.sleep(self.THROTTLE_SECONDS)