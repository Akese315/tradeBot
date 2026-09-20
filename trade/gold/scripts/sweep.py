"""
Balayage automatique de configurations — table complète.

TROIS MODES
-----------
1. UN PARAMÈTRE À LA FOIS (défaut). ~30 variantes, chacune ne diffère de la
   référence que par une valeur. C'est la seule façon d'attribuer un écart à
   une cause précise.

2. RECHERCHE ALÉATOIRE (--random N). Tire N combinaisons au hasard dans
   l'espace de recherche. Sert à explorer les INTERACTIONS entre paramètres,
   que le mode 1 ne peut pas voir (un gros modèle a besoin de plus de
   dropout qu'un petit : tester les deux séparément ne le révèle jamais).

3. VARIANTES D'INDICATEURS (--indicators). Elles modifient IndicatorConfig,
   ce qui oblige à RECONSTRUIRE les features (~1 min par variante). Elles
   sont donc traitées à part.

POURQUOI PAS UNE GRILLE COMPLÈTE
--------------------------------
7 hyperparamètres aux valeurs listées dans SEARCH_SPACE = 24 000
combinaisons. À 2 minutes par run, cela fait 33 jours de calcul non-stop.
Avec les 13 paramètres et 4 valeurs chacun, on dépasse les 250 ans.

La recherche aléatoire trouve en pratique presque aussi bien que la grille
pour une fraction infime du coût (Bergstra & Bengio, 2012). 20 tirages
suffisent généralement à cerner la bonne région.

USAGE
-----
    python scripts/sweep.py --symbol NVDA --list
    python scripts/sweep.py --symbol NVDA --groupe capacite
    python scripts/sweep.py --symbol NVDA --only reference hidden_64
    python scripts/sweep.py --symbol NVDA --random 20
    python scripts/sweep.py --symbol NVDA --indicators
    python scripts/sweep.py --symbol NVDA                    # tout (~30 runs)
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (ASSET_PROFILES, IndicatorConfig,
                             ModelConfig, indicators_for)
from src.ai.features import active_feature_columns, build_feature_frame
from src.data.csv_loader import frame_to_quotes, has_usable_volume, load_ohlcv

# ==========================================================================
# TABLE COMPLÈTE — UN PARAMÈTRE À LA FOIS
# ==========================================================================
GROUPES: dict[str, dict[str, dict]] = {

    # ----------------------------------------------------------------------
    # CAPACITÉ — levier principal contre le sur-apprentissage.
    # À réduire quand la loss de validation remonte pendant que celle
    # d'entraînement continue de descendre.
    # ----------------------------------------------------------------------
    "capacite": {
        "hidden_512": {"hidden_size": 512},
        "hidden_128": {"hidden_size": 128},
        "hidden_64": {"hidden_size": 64},
        "hidden_32": {"hidden_size": 32},
        "hidden_16": {"hidden_size": 16},
        "couches_1": {"num_layers": 1},
        "couches_3": {"num_layers": 3},
    },

    # ----------------------------------------------------------------------
    # RÉGULARISATION — contraindre sans réduire la taille.
    # ----------------------------------------------------------------------
    "regularisation": {
        "dropout_00": {"dropout": 0.0},
        "dropout_01": {"dropout": 0.1},
        "dropout_03": {"dropout": 0.3},
        "dropout_05": {"dropout": 0.5},
        "wd_0": {"weight_decay": 0.0},
        "wd_1e3": {"weight_decay": 1e-3},
        "wd_1e2": {"weight_decay": 1e-2},
    },

    # ----------------------------------------------------------------------
    # CONTEXTE — combien de passé le LSTM voit avant de prédire.
    # ----------------------------------------------------------------------
    "contexte": {
        "fenetre_12": {"sequence_length": 12},
        "fenetre_24": {"sequence_length": 24},
        "fenetre_72": {"sequence_length": 72},
        "fenetre_96": {"sequence_length": 96},
        "fenetre_120": {"sequence_length": 120},
    },

    # ----------------------------------------------------------------------
    # OPTIMISATION — déroulement de la descente de gradient.
    # ----------------------------------------------------------------------
    "optimisation": {
        "lr_1e4": {"learning_rate": 1e-4},
        "lr_3e4": {"learning_rate": 3e-4},
        "lr_3e3": {"learning_rate": 3e-3},
        "batch_16": {"batch_size": 16},
        "batch_32": {"batch_size": 32},
        "batch_128": {"batch_size": 128},
        "batch_256": {"batch_size": 256},
        "patience_15": {"patience": 15},
    },

    # ----------------------------------------------------------------------
    # OBJECTIF — ce que le modèle minimise, et la mise à l'échelle de la
    # cible. ATTENTION : ces variantes changent l'ÉCHELLE de la loss. Ne
    # comparez leur colonne `test` qu'à leur propre `baseline`.
    # ----------------------------------------------------------------------
    "objectif": {
        "perte_mse": {"loss_name": "mse"},
        "perte_l1": {"loss_name": "l1"},
        "echelle_fixe": {"rolling_target_scale": False},
    },

    # ----------------------------------------------------------------------
    # DÉCOUPAGE — répartition train/val/test. Toujours chronologique.
    # ----------------------------------------------------------------------
    "decoupage": {
        "split_60_20": {"train_ratio": 0.60, "val_ratio": 0.20},
        "split_80_10": {"train_ratio": 0.80, "val_ratio": 0.10},
    },
}

VARIANTS: dict[str, dict] = {"reference": {}}
for _groupe in GROUPES.values():
    VARIANTS.update(_groupe)


# ==========================================================================
# VARIANTES D'INDICATEURS — reconstruction des features obligatoire
# ==========================================================================
# Changer une période d'indicateur change les FEATURES, pas seulement le
# modèle : il faut repasser sur les 47 000 bougies. D'où la séparation.
INDICATOR_VARIANTS: dict[str, dict] = {
    "ind_reference": {},
    "ind_courts": {"sma_period": 40, "stoch_k_period": 28,
                   "stoch_d_period": 7, "rsi_period": 28, "adx_period": 28},
    # Périodes de Wilder en bougies pures (14 barres) plutôt qu'en jours.
    "ind_wilder": {"stoch_k_period": 14, "stoch_d_period": 3,
                   "rsi_period": 14, "adx_period": 14},
    "ind_longs": {"sma_period": 280, "stoch_k_period": 196,
                  "stoch_d_period": 42, "rsi_period": 196, "adx_period": 196},
    "ind_sma_courte": {"sma_period": 40},
    "ind_sma_longue": {"sma_period": 280},
}


# ==========================================================================
# ESPACE DE RECHERCHE ALÉATOIRE (option --random)
# ==========================================================================
SEARCH_SPACE: dict[str, list] = {
    "hidden_size": [16, 32, 64, 128, 256],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
    "sequence_length": [12, 24, 48, 72, 96],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128],
    "weight_decay": [0.0, 1e-4, 1e-3, 1e-2],
}


# ==========================================================================
# EXÉCUTION
# ==========================================================================
def _train_one(features, name: str, overrides: dict, symbol: str,
               indicator_config: IndicatorConfig, index: int, total: int,
               columns: list | None = None):
    from src.ai.dataset import build_splits
    from src.ai.train import train

    config = replace(ModelConfig(), **overrides)

    print("=" * 74)
    print(f"[{index}/{total}] {name}")
    print(f"    {overrides if overrides else '(référence)'}")
    print("=" * 74)

    splits = build_splits(features, config.sequence_length,
                          config.train_ratio, config.val_ratio,
                          rolling_target_scale=config.rolling_target_scale,
                          feature_columns=columns)

    context = {"symbole": symbol, "variante": name,
               "modifications": str(overrides) if overrides else "aucune",
               "lignes_features": len(features),
               "fenetres_train": len(splits.train)}

    # save=False : on garde les métriques, pas les poids. Un balayage de 30
    # variantes remplirait models/ de fichiers dont 29 seraient jetés.
    report = train(splits, config, run_name=f"{symbol}_{name}", save=False,
                   indicator_config=indicator_config, context=context)

    return {
        "nom": name,
        "test": report["test_loss"],
        "baseline": report["baseline_loss_zero_prediction"],
        "ecart": report["test_loss"] - report["baseline_loss_zero_prediction"],
        "direction": report["test_direction_accuracy"],
        "epoque": min(report["history"], key=lambda h: h["val_loss"])["epoch"],
    }


def _summary(results: list[dict]) -> None:
    print("=" * 74)
    print("RÉCAPITULATIF — trié par écart à la baseline (le plus bas d'abord)")
    print("=" * 74)
    print(f"{'variante':<18}{'test':>10}{'baseline':>11}{'écart':>10}"
          f"{'direction':>11}{'meill.ép.':>10}")
    print("-" * 74)
    for row in sorted(results, key=lambda r: r["ecart"]):
        flag = " *" if row["ecart"] < 0 else ""
        print(f"{row['nom']:<18}{row['test']:>10.5f}{row['baseline']:>11.5f}"
              f"{row['ecart']:>10.5f}{row['direction']:>10.2%}"
              f"{row['epoque']:>10}{flag}")
    print("-" * 74)
    print("* = bat la baseline (écart négatif)")
    print("\nRAPPEL 1 : sur ~7000 échantillons, l'écart-type de la précision")
    print("directionnelle vaut 0,6 %. Entre 48,8 % et 51,2 % = hasard.")
    print("RAPPEL 2 : les variantes du groupe 'objectif' changent l'échelle")
    print("de la loss. Seule la colonne 'écart' reste comparable.")
    print("\nDétail complet dans historique_runs.txt")


def _load(symbol: str, csv_dir: str, asset: str):
    """Chargement commun : données, profil, détection du volume."""
    profile = ASSET_PROFILES[asset]
    print(f"Profil : {profile.label} — {profile.bars_per_day} bougies/jour")
    source = Path(csv_dir)
    pattern = f"{symbol}_*.csv" if source.is_dir() else "*.csv"
    quotes = frame_to_quotes(load_ohlcv(source, pattern))
    print(f"{len(quotes)} bougies valides")
    use_volume = profile.has_volume and has_usable_volume(quotes)
    columns = active_feature_columns(use_volume)
    print(f"{len(columns)} features actives")
    return quotes, profile, columns, use_volume


def run(symbol: str, csv_dir: str, names: list[str], asset: str,
        overrides_map: dict) -> None:
    print("Chargement des données...")
    quotes, profile, columns, use_volume = _load(symbol, csv_dir, asset)
    indicator_config = indicators_for(profile)
    # Construites UNE SEULE FOIS et partagées par toutes les variantes :
    # c'est ce qui rend le balayage supportable.
    features = build_feature_frame(quotes, indicator_config,
                                   use_volume=use_volume)
    print(f"{len(features)} lignes de features\n")

    results = []
    for i, name in enumerate(names, start=1):
        results.append(_train_one(features, name, overrides_map[name], symbol,
                                  indicator_config, i, len(names), columns))
        print()
    _summary(results)


def run_indicators(symbol: str, csv_dir: str, asset: str) -> None:
    """Chaque variante d'indicateur impose sa propre reconstruction."""
    print("Chargement des données...")
    quotes, profile, columns, use_volume = _load(symbol, csv_dir, asset)
    base = indicators_for(profile)
    print()

    results = []
    total = len(INDICATOR_VARIANTS)
    for i, (name, overrides) in enumerate(INDICATOR_VARIANTS.items(), start=1):
        indicator_config = replace(base, **overrides)
        print(f"--- Reconstruction des features pour {name} ---")
        features = build_feature_frame(quotes, indicator_config,
                                       use_volume=use_volume)
        print(f"{len(features)} lignes\n")
        results.append(_train_one(features, name, {}, symbol,
                                  indicator_config, i, total, columns))
        print()
    _summary(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Balayage de configurations")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--csv-dir", default="data/dataframe")
    parser.add_argument("--asset", default="stock_us", choices=list(ASSET_PROFILES))
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--groupe", default=None, choices=list(GROUPES))
    parser.add_argument("--random", type=int, default=0,
                        help="Nombre de combinaisons aléatoires à tester")
    parser.add_argument("--seed", type=int, default=0,
                        help="Graine du tirage (reproductibilité)")
    parser.add_argument("--indicators", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"{len(VARIANTS)} variantes 'un paramètre à la fois' :\n")
        print("  reference          (référence)")
        for groupe, variantes in GROUPES.items():
            print(f"\n  [{groupe}]")
            for name, ov in variantes.items():
                print(f"    {name:<18} {ov}")
        print(f"\n{len(INDICATOR_VARIANTS)} variantes d'indicateurs "
              f"(option --indicators) :")
        for name, ov in INDICATOR_VARIANTS.items():
            print(f"    {name:<18} {ov if ov else '(référence)'}")
        print("\nEspace de recherche aléatoire (option --random N) :")
        combos = 1
        for key, values in SEARCH_SPACE.items():
            print(f"    {key:<18} {values}")
            combos *= len(values)
        print(f"\n  -> {combos} combinaisons possibles.")
        print(f"     Une grille exhaustive prendrait environ "
              f"{combos * 2 / 60 / 24:.0f} jours de calcul.")
        print("     D'où l'intérêt de --random 20.")
        return

    if args.indicators:
        run_indicators(args.symbol, args.csv_dir, args.asset)
        return

    if args.random > 0:
        # Graine fixée : le même --seed retire les mêmes combinaisons, donc
        # un balayage aléatoire reste reproductible.
        rng = random.Random(args.seed)
        overrides_map = {"reference": {}}
        for i in range(args.random):
            overrides_map[f"alea_{i:02d}"] = {
                key: rng.choice(values) for key, values in SEARCH_SPACE.items()
            }
        run(args.symbol, args.csv_dir, list(overrides_map),
            args.asset, overrides_map)
        return

    if args.groupe:
        names = ["reference"] + list(GROUPES[args.groupe])
    elif args.only:
        unknown = [n for n in args.only if n not in VARIANTS]
        if unknown:
            sys.exit(f"Variantes inconnues : {unknown}")
        names = args.only
    else:
        names = list(VARIANTS)

    run(args.symbol, args.csv_dir, names, args.asset, VARIANTS)


if __name__ == "__main__":
    main()
