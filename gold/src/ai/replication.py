"""
Analyse statistique de runs répliqués.

POURQUOI CE FICHIER EXISTE
--------------------------
Un run unique ne prouve rien. L'initialisation des poids est aléatoire, donc
deux entraînements identiques donnent des résultats légèrement différents.
Quand l'avantage mesuré (−0,0008 sur la loss) est du même ordre que cette
variabilité, un seul tirage ne permet pas de conclure.

La réplication consiste à relancer la MÊME configuration avec des graines
différentes, puis à regarder la distribution des résultats plutôt qu'un point.

CE QUE L'ON REGARDE
-------------------
1. Combien de runs battent la baseline. 5/5 est convaincant, 3/5 ne l'est pas.
2. L'écart moyen ET son écart-type. Si la moyenne est inférieure à l'écart-type,
   l'effet est noyé dans le bruit.
3. Le test de Student sur l'échantillon apparié : la moyenne des écarts est-elle
   significativement différente de zéro ?

Ce n'est pas de la statistique sophistiquée, mais c'est la différence entre
« j'ai un résultat » et « j'ai eu de la chance ».
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def summarise_replications(reports: List[Dict]) -> Dict:
    """Résume N runs de même configuration, graines différentes."""
    gaps = np.array([r["test_loss"] - r["baseline_loss_zero_prediction"]
                     for r in reports], dtype=np.float64)
    directions = np.array([r["test_direction_accuracy"] for r in reports],
                          dtype=np.float64)
    beats = int(sum(1 for g in gaps if g < 0))
    n = len(gaps)

    # Test de Student sur un échantillon : la moyenne des écarts diffère-t-elle
    # de zéro ? `ddof=1` car on estime l'écart-type sur un échantillon.
    mean_gap = float(gaps.mean())
    std_gap = float(gaps.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and std_gap > 0:
        t_stat = mean_gap / (std_gap / math.sqrt(n))
    else:
        t_stat = 0.0

    return {
        "n_runs": n,
        "bat_baseline": f"{beats}/{n}",
        "ecart_moyen": mean_gap,
        "ecart_ecart_type": std_gap,
        "t_statistique": t_stat,
        "direction_moyenne": float(directions.mean()),
        "direction_ecart_type": float(directions.std(ddof=1)) if n > 1 else 0.0,
        "direction_min": float(directions.min()),
        "direction_max": float(directions.max()),
    }


def print_replication_report(summary: Dict, n_test_samples: int | None = None) -> None:
    """Affiche le verdict en clair."""
    n = summary["n_runs"]
    print("=" * 70)
    print(f"RÉPLICATION — {n} runs de configuration identique")
    print("=" * 70)
    print(f"  Battent la baseline        : {summary['bat_baseline']}")
    print(f"  Écart moyen à la baseline  : {summary['ecart_moyen']:+.6f}")
    print(f"  Écart-type entre runs      :  {summary['ecart_ecart_type']:.6f}")
    print(f"  Direction moyenne          : {summary['direction_moyenne']:.2%} "
          f"(± {summary['direction_ecart_type']:.2%})")
    print(f"  Direction min / max        : {summary['direction_min']:.2%} "
          f"/ {summary['direction_max']:.2%}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("-" * 70)
    t = summary["t_statistique"]
    beats, total = summary["bat_baseline"].split("/")
    all_beat = beats == total

    if n < 3:
        print("  Trop peu de runs pour conclure. Visez au moins 5.")
    elif all_beat and t < -2.5:
        print("  VERDICT : effet cohérent et statistiquement net.")
        print("  Tous les runs battent la baseline, et l'écart moyen est")
        print(f"  {abs(t):.1f} fois son erreur type. Ce n'est pas du bruit.")
    elif all_beat:
        print("  VERDICT : effet plausible mais faible.")
        print("  Tous les runs battent la baseline, mais l'écart reste petit")
        print("  devant la variabilité. À confirmer sur plus de runs.")
    elif int(beats) >= n * 0.6:
        print("  VERDICT : indécis. La majorité des runs battent la baseline,")
        print("  mais pas tous. Un effet réel devrait être plus régulier.")
    else:
        print("  VERDICT : pas d'effet démontrable. Le résultat initial était")
        print("  très probablement dû au hasard de l'initialisation.")

    # ------------------------------------------------------------------
    # Significativité de la précision directionnelle
    # ------------------------------------------------------------------
    if n_test_samples:
        sigma = math.sqrt(0.25 / n_test_samples)
        z = (summary["direction_moyenne"] - 0.5) / sigma
        print("-" * 70)
        print(f"  Sur {n_test_samples} échantillons de test, l'écart-type du")
        print(f"  hasard vaut {sigma:.2%}. Votre direction moyenne est à")
        print(f"  {z:.1f} écarts-types de 50 %.")
        if abs(z) < 2:
            print("  -> indiscernable du hasard.")
        elif abs(z) < 3:
            print("  -> significatif, mais à confirmer.")
        else:
            print("  -> significatif.")

    print("=" * 70)
    print("\nRAPPEL ÉCONOMIQUE : un avantage directionnel ne devient une")
    print("stratégie que s'il dépasse les coûts. Espérance par trade ≈")
    print("(précision − 50 %) × 2 × mouvement moyen. Comparez au spread")
    print("aller-retour de votre courtier avant toute conclusion.")