"""
Entraînement DIRECT depuis les CSV bruts, sans passer par MySQL.

C'est le chemin le plus court pour tester la chaîne complète : vous avez déjà
des fichiers dans l'ancien dossier `dataframe/` (NVDA_2010-01.csv, etc.), donc
inutile de re-télécharger quoi que ce soit ni de monter une base.

USAGE
-----
    python scripts/train_from_csv.py --symbol NVDA --csv-dir ../dataframe
    python scripts/train_from_csv.py --symbol NVDA --csv-dir ../dataframe --dry-run

`--dry-run` : construit les features et affiche les statistiques, mais
n'entraîne pas. À FAIRE EN PREMIER — si les statistiques sont aberrantes
(écart-type nul, rendements à 300 %), inutile de lancer l'entraînement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permet de lancer le script depuis n'importe où sans installer le package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import IndicatorConfig, ModelConfig
from src.ai.features import (FEATURE_COLUMNS, TARGET_COLUMN,
                             build_feature_frame)
from src.core.entities import Quote


def load_csv_directory(csv_dir: Path, symbol: str) -> pd.DataFrame:
    """Charge et concatène tous les CSV d'un symbole, triés par date.

    ⚠️ Le tri est INDISPENSABLE : `glob` renvoie les fichiers dans un ordre
    dépendant du système de fichiers. L'ancien code concaténait puis triait
    par la colonne `time`, ce qui est correct — on garde ce filet de sécurité
    en plus du tri des noms de fichiers.
    """
    files = sorted(csv_dir.glob(f"{symbol}_*.csv"))
    if not files:
        sys.exit(f"Aucun fichier {symbol}_*.csv trouvé dans {csv_dir.resolve()}")

    print(f"{len(files)} fichiers trouvés ({files[0].name} → {files[-1].name})")
    frame = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"])

    # Dédoublonnage : les mois se chevauchent parfois d'une bougie chez
    # AlphaVantage. Une bougie en double fausse tous les indicateurs.
    before = len(frame)
    frame = frame.drop_duplicates(subset="time").sort_values("time")
    if before != len(frame):
        print(f"  {before - len(frame)} doublons supprimés")
    return frame.reset_index(drop=True)


def frame_to_quotes(frame: pd.DataFrame) -> list[Quote]:
    """DataFrame -> liste de Quote, en filtrant les lignes invalides."""
    quotes = []
    skipped = 0
    for row in frame.itertuples(index=False):
        try:
            quote = Quote(
                time=row.time.to_pydatetime(),
                open_price=float(row.openPrice),
                high_price=float(row.highPrice),
                low_price=float(row.lowPrice),
                close_price=float(row.closePrice),
                volume=float(row.volume),
            )
        except (ValueError, TypeError, AttributeError):
            skipped += 1
            continue

        # Contrôle de cohérence : le high doit dominer, le low doit être en bas.
        # Une ligne qui viole ça est corrompue et empoisonnerait le stochastique.
        if not (quote.low_price <= quote.close_price <= quote.high_price):
            skipped += 1
            continue
        if quote.close_price <= 0:
            skipped += 1
            continue
        quotes.append(quote)

    if skipped:
        print(f"  {skipped} lignes écartées (incohérentes ou illisibles)")
    return quotes


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement depuis CSV")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--csv-dir", default="dataframe",
                        help="Dossier contenant les CSV bruts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse les données sans entraîner")
    args = parser.parse_args()

    frame = load_csv_directory(Path(args.csv_dir), args.symbol)
    quotes = frame_to_quotes(frame)
    print(f"{len(quotes)} bougies valides, de {quotes[0].time} à {quotes[-1].time}")

    features = build_feature_frame(quotes, IndicatorConfig())

    # ------------------------------------------------------------------
    # CONTRÔLE QUALITÉ — à lire avant tout entraînement
    # ------------------------------------------------------------------
    print("\n===== STATISTIQUES DES FEATURES =====")
    print(features[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T
          [["mean", "std", "min", "max"]].round(4).to_string())

    target = features[TARGET_COLUMN]
    print(f"\nCible ({TARGET_COLUMN}) :")
    print(f"  moyenne          : {target.mean():.6f}  (doit être ≈ 0)")
    print(f"  écart-type       : {target.std():.6f}")
    print(f"  part de hausses  : {(target > 0).mean():.2%}  (doit être ≈ 50 %)")

    # Une corrélation feature/cible > 0,3 sur des rendements horaires est
    # ANORMALE : c'est presque toujours une fuite de données, pas un signal.
    correlations = features[FEATURE_COLUMNS].corrwith(target).abs().sort_values(
        ascending=False)
    print("\nCorrélations |feature ↔ cible| (les plus fortes) :")
    print(correlations.head(5).round(4).to_string())
    if correlations.max() > 0.3:
        print("⚠️ Corrélation suspecte (> 0,30) : cherchez une fuite de données "
              "avant de vous réjouir.")

    if args.dry_run:
        print("\n--dry-run : arrêt avant entraînement.")
        return

    # Import tardif : permet de lancer --dry-run sans avoir installé PyTorch.
    from src.ai.dataset import build_splits
    from src.ai.train import train

    config = ModelConfig()
    splits = build_splits(features, config.sequence_length,
                          config.train_ratio, config.val_ratio,
                          rolling_target_scale=config.rolling_target_scale)
    print(f"\nFenêtres — train {len(splits.train)} | val {len(splits.validation)} "
          f"| test {len(splits.test)}")

    # Contexte consigné dans historique_runs.txt : sans ces informations,
    # deux runs aux mêmes hyperparamètres mais sur des données différentes
    # seraient indiscernables dans le journal.
    context = {
        "symbole": args.symbol,
        "source": str(Path(args.csv_dir)),
        "bougies_valides": len(quotes),
        "periode": f"{quotes[0].time} -> {quotes[-1].time}",
        "lignes_features": len(features),
        "fenetres_train": len(splits.train),
        "fenetres_val": len(splits.validation),
        "fenetres_test": len(splits.test),
        "correlation_max": round(float(correlations.max()), 4),
        "feature_plus_correlee": correlations.index[0],
        "part_de_hausses": f"{(target > 0).mean():.2%}",
    }

    train(splits, config, run_name=f"{args.symbol}_csv",
          indicator_config=IndicatorConfig(), context=context)


if __name__ == "__main__":
    main()