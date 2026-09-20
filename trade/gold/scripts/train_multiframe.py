"""
Entraînement multi-timeframe.

On prédit sur UN timeframe (le base) en s'aidant de plusieurs autres comme
contexte. Le modèle voit, pour chaque bougie du base, l'état simultané des
timeframes supérieurs.

USAGE
-----
    # H1 prédit, avec contexte 4 h + journalier + hebdomadaire
    python scripts/train_multiframe.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 1h --context 4h 1d 1w

    # Analyse seule, sans entraînement
    python scripts/train_multiframe.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 1h --context 4h 1d --dry-run

    # Base plus rapide, contexte plus large
    python scripts/train_multiframe.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 15m --context 1h 4h 1d

CONVENTION DE NOM DE FICHIER
----------------------------
Le script cherche `{SYMBOLE}_{TIMEFRAME}_data.csv` dans `--data-dir`,
c'est-à-dire la convention de vos fichiers : XAU_1h_data.csv, XAU_4h_data.csv…
Si vos noms diffèrent, adaptez `FILE_PATTERN` ci-dessous.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config.settings import ASSET_PROFILES, ModelConfig, indicators_for
from src.ai.features import (TARGET_COLUMN, active_feature_columns,
                             build_feature_frame)
from src.ai.multiframe import (TIMEFRAME_MINUTES, attach_context,
                               build_context_features, verify_no_lookahead)
from src.data.csv_loader import frame_to_quotes, has_usable_volume, load_ohlcv

FILE_PATTERN = "{symbol}_{timeframe}_data.csv"


def load_timeframe(data_dir: Path, symbol: str, timeframe: str):
    path = data_dir / FILE_PATTERN.format(symbol=symbol, timeframe=timeframe)
    if not path.exists():
        available = sorted(p.name for p in data_dir.glob(f"{symbol}_*"))
        sys.exit(f"Fichier introuvable : {path}\nDisponibles : {available}")
    return frame_to_quotes(load_ohlcv(path), verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement multi-timeframe")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--asset", default="gold", choices=list(ASSET_PROFILES))
    parser.add_argument("--base", default="1h", choices=list(TIMEFRAME_MINUTES),
                        help="Timeframe sur lequel on prédit")
    parser.add_argument("--context", nargs="*", default=["4h", "1d"],
                        choices=list(TIMEFRAME_MINUTES),
                        help="Timeframes de contexte")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0,
                        help="Graine aléatoire (reproductibilité)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Nombre de répliques avec des graines différentes. "
                             "5 est un bon minimum pour conclure.")
    args = parser.parse_args()

    base_minutes = TIMEFRAME_MINUTES[args.base]
    for timeframe in args.context:
        if TIMEFRAME_MINUTES[timeframe] <= base_minutes:
            sys.exit(f"Le contexte {timeframe} n'est pas plus lent que la base "
                     f"{args.base}. Un contexte doit couvrir une période plus "
                     "large, sinon il n'apporte aucune information nouvelle.")

    profile = ASSET_PROFILES[args.asset]
    data_dir = Path(args.data_dir)

    print(f"Profil  : {profile.label}")
    print(f"Base    : {args.base}")
    print(f"Contexte: {', '.join(args.context) if args.context else 'aucun'}\n")

    # ------------------------------------------------------------------
    # Timeframe de base
    # ------------------------------------------------------------------
    base_quotes = load_timeframe(data_dir, args.symbol, args.base)
    print(f"Base {args.base} : {len(base_quotes)} bougies, "
          f"de {base_quotes[0].time} à {base_quotes[-1].time}")

    use_volume = profile.has_volume and has_usable_volume(base_quotes)
    base_columns = active_feature_columns(use_volume)

    # Le rythme du profil vaut pour du H1. Sur un autre base, on convertit :
    # 23 bougies/jour en H1 font ~6 bougies/jour en 4 h.
    bars_per_day = max(1, round(profile.bars_per_day * 60 / base_minutes))
    from config.settings import AssetProfile
    scaled = AssetProfile(profile.label, bars_per_day, profile.has_volume)
    indicator_config = indicators_for(scaled)
    print(f"  {bars_per_day} bougies/jour -> SMA {indicator_config.sma_period} bougies")

    base_frame = build_feature_frame(base_quotes, indicator_config,
                                     use_volume=use_volume)
    print(f"  {len(base_frame)} lignes de base\n")

    # ------------------------------------------------------------------
    # Timeframes de contexte
    # ------------------------------------------------------------------
    contexts = {}
    for timeframe in args.context:
        quotes = load_timeframe(data_dir, args.symbol, timeframe)
        context = build_context_features(quotes, timeframe)
        # Contrôle anti-fuite : peu coûteux, et il attrape l'erreur la plus
        # coûteuse qu'on puisse commettre en multi-timeframe.
        verify_no_lookahead(base_frame["time"], context, timeframe)
        contexts[timeframe] = context

    if contexts:
        print("Alignement des contextes :")
        frame, context_columns = attach_context(base_frame, contexts)
        print("  contrôle anti-fuite : OK\n")
    else:
        frame, context_columns = base_frame, []

    columns = base_columns + context_columns
    print(f"{len(frame)} lignes finales, {len(columns)} features "
          f"({len(base_columns)} base + {len(context_columns)} contexte)\n")

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------
    print("===== CORRÉLATIONS |feature ↔ cible| =====")
    target = frame[TARGET_COLUMN]
    correlations = frame[columns].corrwith(target).abs().sort_values(
        ascending=False)
    print(correlations.head(8).round(4).to_string())

    print(f"\nCible : moyenne {target.mean():.6f} | "
          f"écart-type {target.std():.6f} | "
          f"hausses {(target > 0).mean():.2%}")

    if correlations.max() > 0.30:
        print("\n⚠️ Corrélation > 0,30 : cherchez une fuite avant d'entraîner.")

    if args.dry_run:
        print("\n--dry-run : arrêt avant entraînement.")
        return

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    from dataclasses import replace as dc_replace

    from src.ai.dataset import build_splits
    from src.ai.replication import (print_replication_report,
                                    summarise_replications)
    from src.ai.train import train

    base_config = ModelConfig()
    # Les splits ne dépendent PAS de la graine : mêmes données pour toutes les
    # répliques. Seule l'initialisation du réseau varie, ce qui isole
    # exactement la source de variabilité que l'on veut mesurer.
    splits = build_splits(frame, base_config.sequence_length,
                          base_config.train_ratio, base_config.val_ratio,
                          rolling_target_scale=base_config.rolling_target_scale,
                          feature_columns=columns)
    print(f"\nFenêtres — train {len(splits.train)} | "
          f"val {len(splits.validation)} | test {len(splits.test)}")

    context_label = "+".join(args.context) if args.context else "seul"
    reports = []

    for run_index in range(args.repeat):
        seed = args.seed + run_index
        config = dc_replace(base_config, seed=seed)

        if args.repeat > 1:
            print(f"\n{'=' * 70}")
            print(f"RÉPLIQUE {run_index + 1}/{args.repeat} — graine {seed}")
            print("=" * 70)

        report = train(
            splits, config,
            run_name=f"{args.symbol}_{args.base}_ctx_{context_label}_s{seed}",
            # On ne sauvegarde les poids que sur un run unique : garder
            # 5 modèles dont 4 seront jetés n'a pas d'intérêt.
            save=(args.repeat == 1),
            indicator_config=indicator_config,
            context={
                "symbole": args.symbol,
                "profil_actif": args.asset,
                "timeframe_base": args.base,
                "timeframes_contexte": context_label,
                "features_base": len(base_columns),
                "features_contexte": len(context_columns),
                "lignes": len(frame),
                "fenetres_test": len(splits.test),
                "graine": seed,
            })
        reports.append(report)

    if args.repeat > 1:
        print()
        print_replication_report(summarise_replications(reports),
                                 n_test_samples=len(splits.test))


if __name__ == "__main__":
    main()