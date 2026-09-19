"""
Validation walk-forward (réentraînement glissant).

LE PROBLÈME QUE ÇA RÉSOUT
-------------------------
Un découpage unique train/val/test donne UNE mesure. Si elle est bonne, on ne
sait pas si c'est parce que le modèle marche ou parce que cette période-là lui
convenait. Et quand on a testé 40 configurations avant d'en retenir une, le
maximum attendu par pur hasard tourne autour de 2,4 écarts-types : un résultat
à 2,6 σ n'est plus vraiment concluant.

Le walk-forward découpe l'historique en plis successifs :

    pli 1 : train [2004 → 2010]  val [2010 → 2011]  test [2011 → 2012]
    pli 2 : train [2004 → 2011]  val [2011 → 2012]  test [2012 → 2013]
    pli 3 : train [2004 → 2012]  val [2012 → 2013]  test [2013 → 2014]
    ...

Chaque pli entraîne un modèle NEUF et le teste sur une période qu'il n'a
jamais vue. On obtient donc une dizaine de mesures quasi indépendantes au lieu
d'une seule, et surtout on voit si l'avantage est STABLE dans le temps ou
concentré sur une période chanceuse.

C'est le protocole standard avant tout backtest sérieux, et c'est aussi la
réponse directe à la question du changement de régime : le modèle se
réentraîne au fil du temps au lieu de rester figé sur 2004-2019.

DEUX MODES
----------
- `expanding` (ancré) : le train part toujours du début et s'allonge.
  Utilise toute l'histoire. Convient si les régimes anciens restent pertinents.
- `rolling` (glissant) : le train garde une longueur fixe et avance.
  Oublie le passé lointain. Convient si le marché a structurellement changé —
  ce qui est probablement votre cas.

LE PIÈGE ÉVITÉ
--------------
Chaque pli ajuste son PROPRE scaler sur son propre train. Réutiliser un scaler
global ferait entrer des statistiques du futur dans tous les plis, et la
validation perdrait tout son sens — c'est l'erreur la plus fréquente dans les
implémentations maison de walk-forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Bornes d'un pli. Strictement chronologiques : train < val < test."""
    index: int
    train_start: int
    train_end: int
    val_end: int
    test_end: int

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.val_end


def make_folds(n_rows: int, n_folds: int = 8,
               initial_train_ratio: float = 0.40,
               mode: str = "expanding",
               val_fraction: float = 0.5) -> List[Fold]:
    """Construit les plis du walk-forward.

    `initial_train_ratio` : part de l'historique réservée au premier train.
    `val_fraction`        : taille de la validation, en fraction d'un pas de
                            test. 0.5 = validation deux fois plus courte que
                            le test.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError("mode doit valoir 'expanding' ou 'rolling'")

    initial_train = int(n_rows * initial_train_ratio)
    remaining = n_rows - initial_train
    if remaining <= 0:
        raise ValueError("initial_train_ratio trop élevé : rien à tester.")

    step = remaining // n_folds
    val_size = max(1, int(step * val_fraction))

    if step <= val_size:
        raise ValueError(
            f"Pas de test trop court ({step} lignes) pour {n_folds} plis. "
            "Réduisez n_folds ou initial_train_ratio.")

    folds: List[Fold] = []
    for i in range(n_folds):
        test_start = initial_train + i * step
        test_end = test_start + step if i < n_folds - 1 else n_rows

        val_end = test_start
        train_end = val_end - val_size

        if mode == "expanding":
            train_start = 0
        else:
            # Fenêtre glissante : longueur de train constante, égale à celle
            # du premier pli. Le modèle oublie donc le passé lointain.
            train_start = max(0, train_end - (initial_train - val_size))

        if train_end - train_start < 100:
            continue  # pli dégénéré, on le saute

        folds.append(Fold(i + 1, train_start, train_end, val_end, test_end))

    if not folds:
        raise ValueError("Aucun pli exploitable. Vérifiez les paramètres.")
    return folds


def describe_folds(folds: List[Fold], times: Optional[pd.Series] = None) -> None:
    """Affiche le plan de validation avant de lancer les entraînements."""
    print(f"{'pli':>4}  {'train':>16}  {'val':>13}  {'test':>13}")
    print("-" * 56)
    for fold in folds:
        if times is not None:
            start = times.iloc[fold.train_start].strftime("%Y-%m")
            t_end = times.iloc[fold.train_end - 1].strftime("%Y-%m")
            v_end = times.iloc[fold.val_end - 1].strftime("%Y-%m")
            e_end = times.iloc[fold.test_end - 1].strftime("%Y-%m")
            print(f"{fold.index:>4}  {start}→{t_end}  "
                  f"→{v_end}  →{e_end}  ({fold.n_test} lignes)")
        else:
            print(f"{fold.index:>4}  [{fold.train_start:>6}:{fold.train_end:>6}]"
                  f"  [:{fold.val_end:>6}]  [:{fold.test_end:>6}]")
    print()


def _binomial_tail(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) pour X ~ Binomiale(n, p). Calcul exact, sans scipy."""
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def summarise_folds(results: List[Dict]) -> Dict:
    """Agrège les résultats de tous les plis."""
    gaps = np.array([r["gap"] for r in results], dtype=np.float64)
    directions = np.array([r["direction"] for r in results], dtype=np.float64)
    sizes = np.array([r["n_test"] for r in results], dtype=np.float64)

    n = len(results)
    beats = int((gaps < 0).sum())
    above_half = int((directions > 0.5).sum())

    # Direction globale : moyenne PONDÉRÉE par la taille de chaque pli.
    # Une moyenne simple donnerait autant de poids à un pli de 200 lignes
    # qu'à un pli de 2000.
    pooled_direction = float((directions * sizes).sum() / sizes.sum())
    pooled_n = int(sizes.sum())
    sigma = math.sqrt(0.25 / pooled_n)
    z = (pooled_direction - 0.5) / sigma

    mean_gap = float(gaps.mean())
    std_gap = float(gaps.std(ddof=1)) if n > 1 else 0.0
    t_stat = mean_gap / (std_gap / math.sqrt(n)) if n > 1 and std_gap > 0 else 0.0

    return {
        "n_folds": n,
        "plis_battant_baseline": beats,
        "plis_direction_sup_50": above_half,
        "p_value_signe": _binomial_tail(beats, n),
        "ecart_moyen": mean_gap,
        "ecart_ecart_type": std_gap,
        "t_statistique": t_stat,
        "direction_ponderee": pooled_direction,
        "direction_min": float(directions.min()),
        "direction_max": float(directions.max()),
        "echantillons_test_total": pooled_n,
        "z_direction": z,
    }


def print_fold_report(results: List[Dict], summary: Dict) -> None:
    """Rapport détaillé, pli par pli puis agrégé."""
    print("=" * 72)
    print("WALK-FORWARD — résultats par pli")
    print("=" * 72)
    print(f"{'pli':>4} {'période test':>18} {'test':>10} {'baseline':>10}"
          f" {'écart':>10} {'direction':>10}")
    print("-" * 72)
    for r in results:
        flag = " *" if r["gap"] < 0 else ""
        print(f"{r['fold']:>4} {r.get('periode', ''):>18} {r['test_loss']:>10.5f}"
              f" {r['baseline']:>10.5f} {r['gap']:>10.5f}"
              f" {r['direction']:>9.2%}{flag}")

    print("=" * 72)
    n = summary["n_folds"]
    print(f"  Plis battant la baseline   : {summary['plis_battant_baseline']}/{n}"
          f"   (p = {summary['p_value_signe']:.3f} sous H0)")
    print(f"  Plis avec direction > 50 % : {summary['plis_direction_sup_50']}/{n}")
    print(f"  Écart moyen                : {summary['ecart_moyen']:+.6f}"
          f"  (± {summary['ecart_ecart_type']:.6f})")
    print(f"  Direction pondérée         : {summary['direction_ponderee']:.2%}"
          f"  sur {summary['echantillons_test_total']} échantillons")
    print(f"  Direction min / max        : {summary['direction_min']:.2%}"
          f" / {summary['direction_max']:.2%}")
    print(f"  z de la direction          : {summary['z_direction']:.2f}")

    print("-" * 72)
    p_value = summary["p_value_signe"]
    z = summary["z_direction"]

    if p_value < 0.05 and z > 2.5:
        print("  VERDICT : avantage stable dans le temps.")
        print("  L'effet se reproduit sur des périodes indépendantes, ce qui")
        print("  est bien plus convaincant qu'un découpage unique.")
    elif p_value < 0.05 or z > 2.5:
        print("  VERDICT : signal plausible mais irrégulier.")
        print("  Un des deux critères passe, pas l'autre. Regardez la colonne")
        print("  direction pli par pli : l'avantage est-il concentré sur")
        print("  quelques périodes ?")
    else:
        print("  VERDICT : pas d'avantage démontrable sur données glissantes.")
        print("  Le résultat du découpage unique ne se reproduit pas. C'était")
        print("  très probablement un artefact de cette période de test.")

    print("=" * 72)
    print("\nÀ REGARDER : la dispersion entre plis. Un modèle exploitable est")
    print("régulier. Un modèle à 58 % sur deux plis et 45 % sur six autres a")
    print("la même moyenne qu'un modèle à 51 % partout, mais il est")
    print("inutilisable — vous ne saurez jamais dans quel régime vous êtes.")