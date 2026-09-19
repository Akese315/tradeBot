"""
Validation walk-forward en ligne de commande.

Réentraîne un modèle neuf sur chaque pli chronologique et teste sur la période
suivante. Donne une dizaine de mesures quasi indépendantes au lieu d'une.

USAGE
-----
    # Multi-timeframe, fenêtre glissante (recommandé)
    python scripts/walkforward.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 4h --context 1d 1w \\
        --folds 8 --mode rolling

    # Fenêtre ancrée : le train part toujours du début
    python scripts/walkforward.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 4h --context 1d 1w --mode expanding

    # Timeframe unique, sans contexte
    python scripts/walkforward.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 1h

    # Voir le plan de validation sans entraîner
    python scripts/walkforward.py --symbol XAU --asset gold \\
        --data-dir data/gold --base 4h --context 1d --plan-only

DURÉE
-----
Comptez le temps d'un entraînement multiplié par le nombre de plis. Avec 8
plis à ~6 minutes, prévoyez trois quarts d'heure. Les plis initiaux sont plus
rapides (moins de données d'entraînement).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config.settings import (ASSET_PROFILES, AssetProfile, ModelConfig,
                             indicators_for)
from src.ai.features import active_feature_columns, build_feature_frame
from src.ai.multiframe import (TIMEFRAME_MINUTES, attach_context,
                               build_context_features, verify_no_lookahead)
from src.ai.walkforward import (describe_folds, make_folds, print_fold_report,
                                summarise_folds)
from src.data.csv_loader import frame_to_quotes, has_usable_volume, load_ohlcv

FILE_PATTERN = "{symbol}_{timeframe}_data.csv"


def load_timeframe(data_dir: Path, symbol: str, timeframe: str):
    path = data_dir / FILE_PATTERN.format(symbol=symbol, timeframe=timeframe)
    if not path.exists():
        available = sorted(p.name for p in data_dir.glob(f"{symbol}_*"))
        sys.exit(f"Fichier introuvable : {path}\nDisponibles : {available}")
    return frame_to_quotes(load_ohlcv(path), verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation walk-forward")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--asset", default="gold", choices=list(ASSET_PROFILES))
    parser.add_argument("--base", default="4h", choices=list(TIMEFRAME_MINUTES))
    parser.add_argument("--context", nargs="*", default=[],
                        choices=list(TIMEFRAME_MINUTES))
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--mode", default="rolling",
                        choices=["rolling", "expanding"])
    parser.add_argument("--initial-train", type=float, default=0.40,
                        help="Part de l'historique pour le premier train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true",
                        help="Afficher le plan de validation sans entraîner")
    parser.add_argument("--regime", nargs="*", default=None,
                        metavar="COLONNE",
                        help="Analyse conditionnelle par régime. Colonnes "
                             "usuelles : realized_vol adx. Sans argument, "
                             "utilise realized_vol et adx.")
    parser.add_argument("--buckets", type=int, default=5,
                        help="Nombre de tranches de régime")
    parser.add_argument("--cost", type=float, default=0.02,
                        help="Coût aller-retour en pourcentage (spread + frais)")
    args = parser.parse_args()

    base_minutes = TIMEFRAME_MINUTES[args.base]
    for timeframe in args.context:
        if TIMEFRAME_MINUTES[timeframe] <= base_minutes:
            sys.exit(f"Le contexte {timeframe} doit être plus lent que la base "
                     f"{args.base}.")

    profile = ASSET_PROFILES[args.asset]
    data_dir = Path(args.data_dir)

    print(f"Profil   : {profile.label}")
    print(f"Base     : {args.base}")
    print(f"Contexte : {', '.join(args.context) if args.context else 'aucun'}")
    print(f"Mode     : {args.mode} — {args.folds} plis\n")

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------
    base_quotes = load_timeframe(data_dir, args.symbol, args.base)
    print(f"{len(base_quotes)} bougies, de {base_quotes[0].time} "
          f"à {base_quotes[-1].time}")

    use_volume = profile.has_volume and has_usable_volume(base_quotes)
    base_columns = active_feature_columns(use_volume)
    bars_per_day = max(1, round(profile.bars_per_day * 60 / base_minutes))
    scaled = AssetProfile(profile.label, bars_per_day, profile.has_volume)
    indicator_config = indicators_for(scaled)

    frame = build_feature_frame(base_quotes, indicator_config,
                                use_volume=use_volume)

    context_columns: list = []
    if args.context:
        contexts = {}
        for timeframe in args.context:
            quotes = load_timeframe(data_dir, args.symbol, timeframe)
            context = build_context_features(quotes, timeframe)
            verify_no_lookahead(frame["time"], context, timeframe)
            contexts[timeframe] = context
        frame, context_columns = attach_context(frame, contexts, verbose=False)
        print("  contrôle anti-fuite : OK")

    columns = base_columns + context_columns
    print(f"{len(frame)} lignes, {len(columns)} features\n")

    # ------------------------------------------------------------------
    # Plan de validation
    # ------------------------------------------------------------------
    folds = make_folds(len(frame), n_folds=args.folds,
                       initial_train_ratio=args.initial_train,
                       mode=args.mode)
    print("PLAN DE VALIDATION")
    describe_folds(folds, frame["time"])

    if args.plan_only:
        print("--plan-only : arrêt avant entraînement.")
        return

    # ------------------------------------------------------------------
    # Entraînement pli par pli
    # ------------------------------------------------------------------
    from src.ai.dataset import build_splits_from_bounds
    from src.ai.regime import (analyse_by_confidence,
                               analyse_by_confidence_marginal,
                               analyse_by_quantiles, final_verdict,
                               print_confidence_table,
                               print_confidence_verdict, print_marginal_table,
                               print_regime_table, print_stability,
                               test_row_indices)
    from src.ai.train import train

    # `--regime` sans valeur -> régimes par défaut.
    regime_columns = args.regime
    if regime_columns is not None and len(regime_columns) == 0:
        regime_columns = ["realized_vol", "adx"]
    if regime_columns:
        missing = [c for c in regime_columns if c not in frame.columns]
        if missing:
            sys.exit(f"Colonnes de régime absentes : {missing}\n"
                     f"Disponibles : {[c for c in columns]}")

    base_config = ModelConfig()
    context_label = "+".join(args.context) if args.context else "seul"
    results = []
    fold_regimes: dict = {}

    for fold in folds:
        period = (f"{frame['time'].iloc[fold.val_end].strftime('%Y-%m')}"
                  f"→{frame['time'].iloc[fold.test_end - 1].strftime('%Y-%m')}")

        print("=" * 72)
        print(f"PLI {fold.index}/{len(folds)} — test {period} "
              f"({fold.n_train} lignes d'entraînement)")
        print("=" * 72)

        # Chaque pli a son propre scaler, ajusté sur SON train. C'est le
        # point critique : un scaler global ferait fuiter le futur partout.
        splits = build_splits_from_bounds(
            frame, base_config.sequence_length,
            fold.train_start, fold.train_end, fold.val_end, fold.test_end,
            rolling_target_scale=base_config.rolling_target_scale,
            feature_columns=columns)

        if len(splits.test) < 50:
            print(f"  pli ignoré : seulement {len(splits.test)} fenêtres de test\n")
            continue

        report = train(
            splits, dc_replace(base_config, seed=args.seed),
            return_predictions=bool(regime_columns),
            run_name=f"{args.symbol}_{args.base}_wf{fold.index}_{args.mode}",
            save=False,
            indicator_config=indicator_config,
            context={
                "symbole": args.symbol,
                "timeframe_base": args.base,
                "timeframes_contexte": context_label,
                "mode_walkforward": args.mode,
                "pli": f"{fold.index}/{len(folds)}",
                "periode_test": period,
                "lignes_train": fold.n_train,
                "fenetres_test": len(splits.test),
            })

        if regime_columns:
            # Rattachement des prédictions à leurs lignes du DataFrame :
            # c'est ce qui permet de connaître le régime de marché au moment
            # de chaque prédiction.
            indices = test_row_indices(fold.val_end, len(report["predictions"]),
                                       base_config.sequence_length)
            fold_regimes[fold.index] = {
                "indices": indices,
                "predictions": report["predictions"],
                "targets": report["targets"],
            }

        results.append({
            "fold": fold.index,
            "periode": period,
            "test_loss": report["test_loss"],
            "baseline": report["baseline_loss_zero_prediction"],
            "gap": (report["test_loss"]
                    - report["baseline_loss_zero_prediction"]),
            "direction": report["test_direction_accuracy"],
            "n_test": len(splits.test),
        })
        print()

    if len(results) < 2:
        sys.exit("Trop peu de plis exploitables pour conclure.")

    print()
    print_fold_report(results, summarise_folds(results))

    # ------------------------------------------------------------------
    # Analyse par régime
    # ------------------------------------------------------------------
    if not regime_columns or not fold_regimes:
        return

    from src.ai.features import TARGET_COLUMN

    all_indices = np.concatenate([d["indices"] for d in fold_regimes.values()])
    all_predictions = np.concatenate(
        [d["predictions"] for d in fold_regimes.values()])
    all_targets = np.concatenate([d["targets"] for d in fold_regimes.values()])

    for column in regime_columns:
        print("\n" + "=" * 84)
        print(f"ANALYSE PAR RÉGIME — {column}")
        print("=" * 84)

        # Agrégé sur tous les plis.
        pooled = analyse_by_quantiles(
            frame, all_indices, all_predictions, all_targets,
            regime_column=column, target_column=TARGET_COLUMN,
            n_buckets=args.buckets, cost_pct=args.cost)
        print_regime_table(pooled, column, args.cost)

        # Puis pli par pli, pour juger de la stabilité.
        per_fold = {}
        for index, data in fold_regimes.items():
            per_fold[index] = analyse_by_quantiles(
                frame, data["indices"], data["predictions"], data["targets"],
                regime_column=column, target_column=TARGET_COLUMN,
                n_buckets=args.buckets, cost_pct=args.cost)
        print_stability(per_fold, args.buckets)
        final_verdict(pooled, per_fold, args.buckets)

    # ------------------------------------------------------------------
    # Analyse par confiance du modèle — indépendante des régimes marché
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSE PAR CONFIANCE DU MODÈLE")
    print("=" * 80)
    raw_moves = frame[TARGET_COLUMN].to_numpy()[all_indices]
    confidence_rows = analyse_by_confidence(
        all_predictions, all_targets, raw_moves,
        n_buckets=10, cost_pct=args.cost)
    print_confidence_table(confidence_rows, args.cost)
    marginal_rows = analyse_by_confidence_marginal(
        all_predictions, all_targets, raw_moves,
        n_buckets=10, cost_pct=args.cost)
    print_marginal_table(marginal_rows)
    print_confidence_verdict(confidence_rows, marginal_rows, args.cost)


if __name__ == "__main__":
    main()