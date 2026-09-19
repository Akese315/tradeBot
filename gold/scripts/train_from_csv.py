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

from config.settings import ASSET_PROFILES, ModelConfig, indicators_for
from src.ai.features import (TARGET_COLUMN, active_feature_columns,
                             build_feature_frame)
from src.data.csv_loader import frame_to_quotes, has_usable_volume, load_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement depuis CSV")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--csv-dir", default="data/dataframe",
                        help="Dossier OU fichier CSV")
    parser.add_argument("--asset", default="stock_us",
                        choices=list(ASSET_PROFILES),
                        help="Profil de marché : règle le rythme des indicateurs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyse les données sans entraîner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Répliques avec graines différentes")
    args = parser.parse_args()

    profile = ASSET_PROFILES[args.asset]
    print(f"Profil : {profile.label} — {profile.bars_per_day} bougies/jour")
    if profile.note:
        print(f"         {profile.note}")

    source = Path(args.csv_dir)
    pattern = f"{args.symbol}_*.csv" if source.is_dir() else "*.csv"
    frame = load_ohlcv(source, pattern)
    quotes = frame_to_quotes(frame)
    if not quotes:
        sys.exit("Aucune bougie exploitable.")
    print(f"{len(quotes)} bougies valides, de {quotes[0].time} à {quotes[-1].time}")

    # Le volume est détecté, pas supposé : un fichier d'or peut contenir une
    # colonne volume remplie de zéros, ce qui est pire qu'une absence.
    use_volume = profile.has_volume and has_usable_volume(quotes)
    if profile.has_volume and not use_volume:
        print("  ! colonne volume vide ou constante -> feature désactivée")
    columns = active_feature_columns(use_volume)
    print(f"{len(columns)} features actives\n")

    indicator_config = indicators_for(profile)
    features = build_feature_frame(quotes, indicator_config,
                                   use_volume=use_volume)

    # ------------------------------------------------------------------
    # CONTRÔLE QUALITÉ — à lire avant tout entraînement
    # ------------------------------------------------------------------
    print("\n===== STATISTIQUES DES FEATURES =====")
    print(features[columns + [TARGET_COLUMN]].describe().T
          [["mean", "std", "min", "max"]].round(4).to_string())

    target = features[TARGET_COLUMN]
    print(f"\nCible ({TARGET_COLUMN}) :")
    print(f"  moyenne          : {target.mean():.6f}  (doit être ≈ 0)")
    print(f"  écart-type       : {target.std():.6f}")
    print(f"  part de hausses  : {(target > 0).mean():.2%}  (doit être ≈ 50 %)")

    # Une corrélation feature/cible > 0,3 sur des rendements horaires est
    # ANORMALE : c'est presque toujours une fuite de données, pas un signal.
    correlations = features[columns].corrwith(target).abs().sort_values(
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
                          rolling_target_scale=config.rolling_target_scale,
                          feature_columns=columns)
    print(f"\nFenêtres — train {len(splits.train)} | val {len(splits.validation)} "
          f"| test {len(splits.test)}")

    # Contexte consigné dans historique_runs.txt : sans ces informations,
    # deux runs aux mêmes hyperparamètres mais sur des données différentes
    # seraient indiscernables dans le journal.
    context = {
        "symbole": args.symbol,
        "profil_actif": args.asset,
        "bougies_par_jour": profile.bars_per_day,
        "features_actives": len(columns),
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

    from dataclasses import replace as dc_replace

    from src.ai.replication import (print_replication_report,
                                    summarise_replications)

    context["fenetres_test"] = len(splits.test)
    reports = []
    for run_index in range(args.repeat):
        seed = args.seed + run_index
        if args.repeat > 1:
            print(f"\n=== RÉPLIQUE {run_index + 1}/{args.repeat} — graine {seed} ===")
        reports.append(train(
            splits,
            dc_replace(config, seed=seed),
            run_name=f"{args.symbol}_{args.asset}_s{seed}",
            save=(args.repeat == 1),
            indicator_config=indicator_config,
            context={**context, "graine": seed}))

    if args.repeat > 1:
        print()
        print_replication_report(summarise_replications(reports),
                                 n_test_samples=len(splits.test))


if __name__ == "__main__":
    main()