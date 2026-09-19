"""
Bot temps réel : interroge le marché, met à jour les indicateurs, journalise.

⚠️ Le bot NE PASSE AUCUN ORDRE. Il observe et enregistre. Ajouter l'exécution
d'ordres avant d'avoir (a) un modèle qui bat la baseline, (b) un backtest avec
frais et slippage, (c) une gestion du risque, revient à automatiser une perte.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from config.settings import DatabaseConfig, IndicatorConfig
from src.core.entities import Quote
from src.core.indicators import Indicators
from src.data.database import get_connection, insert_live_snapshot
from src.data.providers import FinnhubClient, MarketDataError

# `logging` plutôt que `print` : horodatage, niveaux de gravité, redirection
# vers un fichier. Indispensable dès qu'un processus tourne sans surveillance.
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, symbol: str, client: Optional[FinnhubClient] = None,
                 indicator_config: Optional[IndicatorConfig] = None) -> None:
        self.symbol = symbol
        self.client = client or FinnhubClient()
        self.config = indicator_config or IndicatorConfig()
        self.indicators = Indicators(symbol)
        self._market_was_open: Optional[bool] = None

        # Validation immédiate du symbole : mieux vaut échouer au démarrage
        # qu'après six heures de boucle sur un ticker inexistant.
        if not self.client.symbol_exists(symbol):
            raise MarketDataError(f"Symbole inconnu chez Finnhub : {symbol}")

    def tick(self, conn) -> None:
        """Un cycle : cotation -> indicateurs -> journalisation en base."""
        market_open = self.client.is_market_open()
        if market_open != self._market_was_open:
            # On ne log le changement d'état qu'une fois, pas à chaque boucle
            # (l'ancien code inondait la console).
            logger.info("%s : marché %s", self.symbol,
                        "ouvert" if market_open else "fermé")
            self._market_was_open = market_open
        if not market_open:
            return

        quote: Quote = self.client.get_quote(self.symbol)
        self.indicators.push(
            quote,
            sma_period=self.config.sma_period,
            stoch_k_period=self.config.stoch_k_period,
            stoch_d_period=self.config.stoch_d_period,
            rsi_period=self.config.rsi_period,
            adx_period=self.config.adx_period,
        )

        snap = self.indicators.snapshot()
        if snap is None:
            logger.debug("%s : indicateurs en cours de chauffe", self.symbol)
            return

        insert_live_snapshot(conn, (
            self.symbol,
            quote.close_price,
            0.0,  # percentChange : à calculer si l'API le fournit
            quote.close_price - quote.open_price,
            quote.high_price,
            quote.low_price,
            quote.volume,
            snap["sma"].value,
            snap["ema"].value,
            snap["rsi"].value,
            snap["adx"].value,
            snap["stoch_k"].value,
            snap["stoch_d"].value,
        ))

    def run(self, interval_seconds: int = 60,
            db_config: Optional[DatabaseConfig] = None) -> None:
        """Boucle principale. Ctrl+C pour arrêter proprement."""
        logger.info("Démarrage du bot sur %s (intervalle %ss)",
                    self.symbol, interval_seconds)
        with get_connection(db_config) as conn:
            try:
                while True:
                    try:
                        self.tick(conn)
                    except MarketDataError as error:
                        # Une erreur réseau ne doit pas tuer le bot : on log
                        # et on réessaie au cycle suivant.
                        logger.warning("Erreur de données : %s", error)
                    time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Arrêt demandé par l'utilisateur.")