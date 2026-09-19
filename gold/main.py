"""
Point d'entrée unique en ligne de commande.

⚠️ L'ancien projet avait DEUX `main()` avec deux argparse (`main.py` et
`dataManager.py`), et le README documentait une troisième syntaxe
(`python3 bot.py start ...`) qui n'existait dans aucun fichier. Par ailleurs
`dataManager.main()` appelait `cleanTrainingData(args.symbol, args.year)`
alors que la fonction exigeait 4 arguments : la commande `delete` plantait
systématiquement.

Une seule porte d'entrée = une seule vérité.

USAGE
-----
    python main.py harvest --symbol NVDA --year 2023
    python main.py delete  --symbol NVDA [--year 2023]
    python main.py train   --symbol NVDA
    python main.py run     --symbol NVDA --interval 60
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import IndicatorConfig, ModelConfig
from src.ai.dataset import build_splits
from src.ai.features import build_feature_frame, rows_to_quotes
from src.ai.train import train
from src.bot import TradingBot
from src.data.database import (delete_training_data, fetch_training_data,
                               get_connection, insert_training_rows)
from src.data.providers import AlphaVantageClient


def command_harvest(args: argparse.Namespace) -> None:
    """Télécharge une année de bougies et l'enregistre en base."""
    client = AlphaVantageClient()
    with get_connection() as conn:
        for month, rows in client.iter_year(args.symbol, int(args.year),
                                            args.interval):
            count = insert_training_rows(conn, rows)
            print(f"{month} : {count} lignes insérées/mises à jour")


def command_delete(args: argparse.Namespace) -> None:
    with get_connection() as conn:
        deleted = delete_training_data(conn, args.symbol, args.year)
        print(f"{deleted} lignes supprimées pour {args.symbol}")


def command_train(args: argparse.Namespace) -> None:
    """Chaîne complète : base -> bougies -> features -> découpage -> modèle."""
    with get_connection() as conn:
        rows = fetch_training_data(conn, args.symbol)

    if not rows:
        sys.exit(f"Aucune donnée en base pour {args.symbol}. "
                 f"Lancez d'abord : python main.py harvest --symbol {args.symbol} "
                 f"--year 2023")

    print(f"{len(rows)} bougies chargées depuis la base.")
    quotes = rows_to_quotes(rows)
    frame = build_feature_frame(quotes, IndicatorConfig())

    config = ModelConfig()
    splits = build_splits(frame, config.sequence_length,
                          config.train_ratio, config.val_ratio)
    print(f"Train {len(splits.train)} | Val {len(splits.validation)} "
          f"| Test {len(splits.test)} fenêtres")

    train(splits, config, run_name=f"{args.symbol}_{config.hidden_size}h")


def command_run(args: argparse.Namespace) -> None:
    TradingBot(args.symbol).run(interval_seconds=args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TradeBot — CLI unifiée")
    subparsers = parser.add_subparsers(dest="command", required=True)

    harvest = subparsers.add_parser("harvest", help="Collecte l'historique")
    harvest.add_argument("--symbol", required=True)
    harvest.add_argument("--year", required=True)
    harvest.add_argument("--interval", default="60min",
                         choices=["1min", "5min", "15min", "30min", "60min"])
    harvest.set_defaults(func=command_harvest)

    delete = subparsers.add_parser("delete", help="Supprime des données")
    delete.add_argument("--symbol", required=True)
    delete.add_argument("--year", default=None)
    delete.set_defaults(func=command_delete)

    train_parser = subparsers.add_parser("train", help="Entraîne le modèle")
    train_parser.add_argument("--symbol", required=True)
    train_parser.set_defaults(func=command_train)

    run = subparsers.add_parser("run", help="Lance le bot temps réel")
    run.add_argument("--symbol", required=True)
    run.add_argument("--interval", type=int, default=60,
                     help="Secondes entre deux cycles")
    run.set_defaults(func=command_run)

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    arguments = build_parser().parse_args()
    arguments.func(arguments)
