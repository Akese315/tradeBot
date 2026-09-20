"""
Couche d'accès à la base MySQL.

⚠️ TROIS PROBLÈMES GRAVES CORRIGÉS ICI
--------------------------------------
1. **Connexion à l'import.** L'ancien `bot.py` et `dataManager.py` ouvraient
   une connexion MySQL *au moment de l'import du module*. Conséquence : le
   simple fait de faire `import bot` dans un notebook ouvrait une connexion,
   et `bot.py` la refermait aussitôt en fin de fichier (`cursor.close()` au
   niveau module) — donc `main.py`, qui fait `from bot import *`, héritait
   d'un curseur DÉJÀ FERMÉ. C'est la cause de l'erreur silencieuse au train.

2. **Injection SQL.** Toutes les requêtes étaient construites par
   concaténation de chaînes :
       "... WHERE symbol='" + symbol + "';"
   Un symbole contenant une apostrophe casse la requête ; un symbole
   malveillant l'exécute. Ici on utilise exclusivement des requêtes
   paramétrées (`%s`), où le driver échappe les valeurs.

3. **`min(rows)` sur des tuples.** `rows` est une liste de tuples ; `min()`
   comparait les tuples lexicographiquement, pas les prix numériquement.
   Résultat faux dès que les prix n'ont pas tous le même nombre de chiffres
   ("9.5" > "10.2" en comparaison de chaînes). On agrège maintenant en SQL.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Iterator, List, Sequence, Tuple

import mysql.connector
from mysql.connector.connection import MySQLConnection

from config.settings import DatabaseConfig


@contextmanager
def get_connection(config: DatabaseConfig | None = None) -> Iterator[MySQLConnection]:
    """Ouvre une connexion et GARANTIT sa fermeture, même en cas d'exception.

    Usage :
        with get_connection() as conn:
            rows = fetch_training_data(conn, "NVDA")
    """
    config = config or DatabaseConfig()
    conn = mysql.connector.connect(
        host=config.host,
        user=config.user,
        password=config.password,
        database=config.database,
    )
    try:
        yield conn
    finally:
        conn.close()


# ==========================================================================
# LECTURE
# ==========================================================================
def fetch_training_data(conn: MySQLConnection, symbol: str) -> List[Tuple]:
    """Récupère tout l'historique H1 d'un symbole, trié chronologiquement.

    Le tri ASC est NON NÉGOCIABLE : les indicateurs sont incrémentaux, un
    historique désordonné produirait des valeurs sans aucun sens.
    """
    query = (
        "SELECT time, openPrice, highPrice, lowPrice, closePrice, volume "
        "FROM training_data_hour WHERE symbol = %s ORDER BY time ASC"
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (symbol,))
        return cursor.fetchall()


def fetch_price_extremes(conn: MySQLConnection, symbol: str,
                         start: datetime, end: datetime) -> Tuple[float, float]:
    """Plus bas / plus haut sur une fenêtre. L'agrégation est faite par MySQL :
    c'est plus rapide et cela évite le bug de comparaison de tuples en Python."""
    query = (
        "SELECT MIN(lowPrice), MAX(highPrice) FROM training_data_hour "
        "WHERE symbol = %s AND time > %s AND time <= %s"
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (symbol, start, end))
        low, high = cursor.fetchone()
    return float(low), float(high)


# ==========================================================================
# ÉCRITURE
# ==========================================================================
INSERT_TRAINING_ROW = (
    "INSERT INTO training_data_hour "
    "(time, symbol, openPrice, highPrice, lowPrice, closePrice, volume) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
    # Idempotence : relancer une collecte ne duplique plus les lignes.
    # Nécessite l'index UNIQUE (symbol, time) — voir sql/schema.sql.
    "ON DUPLICATE KEY UPDATE "
    "openPrice = VALUES(openPrice), highPrice = VALUES(highPrice), "
    "lowPrice = VALUES(lowPrice), closePrice = VALUES(closePrice), "
    "volume = VALUES(volume)"
)


def insert_training_rows(conn: MySQLConnection, rows: Sequence[Tuple]) -> int:
    """Insertion par LOT (`executemany`) puis un seul `commit`.

    ⚠️ L'ancien code faisait un `commit()` après CHAQUE ligne, à l'intérieur
    de la boucle : environ 1 700 allers-retours disque par mois de données.
    C'est ce qui rendait `harvestYear` interminable.
    """
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_TRAINING_ROW, rows)
        inserted = cursor.rowcount
    conn.commit()
    return inserted


def delete_training_data(conn: MySQLConnection, symbol: str,
                         year: str | None = None) -> int:
    """Supprime l'historique d'un symbole, éventuellement limité à une année."""
    if year is None:
        query = "DELETE FROM training_data_hour WHERE symbol = %s"
        params: tuple = (symbol,)
    else:
        query = "DELETE FROM training_data_hour WHERE symbol = %s AND YEAR(time) = %s"
        params = (symbol, year)

    with conn.cursor() as cursor:
        cursor.execute(query, params)
        deleted = cursor.rowcount
    conn.commit()
    return deleted


INSERT_LIVE_SNAPSHOT = (
    "INSERT INTO stocks "
    "(symbol, currentPrice, percentChange, changePrice, highPriceDay, "
    " lowPriceDay, volume, sma, ema, rsi, adx, stoch_k, stoch_d) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


def insert_live_snapshot(conn: MySQLConnection, values: Iterable) -> None:
    """Enregistre l'état temps réel du bot (une ligne par tick)."""
    with conn.cursor() as cursor:
        cursor.execute(INSERT_LIVE_SNAPSHOT, tuple(values))
    conn.commit()