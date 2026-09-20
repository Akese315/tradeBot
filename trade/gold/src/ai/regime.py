"""
Analyse conditionnelle par régime de marché.

LA QUESTION POSÉE
-----------------
Le walk-forward a montré un avantage directionnel réel mais IRRÉGULIER :
54,48 % sur 2024-2026, 49,13 % sur 2016-2018. La moyenne de 51,31 % ne
suffit pas pour trader, parce qu'on ne sait pas quand elle s'applique.

Ce module ne cherche pas à améliorer le modèle. Il répond à une question
différente et plus utile : **dans quelles conditions ce modèle fonctionne-t-il
déjà ?**

Si l'avantage se concentre sur les bougies à forte volatilité, alors « ne
trader que quand la volatilité dépasse tel seuil » devient une règle
exploitable — sans changer une ligne du réseau.

LE PIÈGE À CONNAÎTRE
--------------------
Découper les résultats en tranches, c'est faire des comparaisons multiples.
Avec 5 tranches, la meilleure paraîtra bonne même sur du bruit pur : le
maximum attendu par hasard tourne autour de 1,9 écart-type.

Deux garde-fous appliqués ici :
  1. Les seuils viennent des QUANTILES du régime, pas d'un seuil choisi après
     avoir vu les résultats. On ne fait pas défiler 50 valeurs pour retenir
     la plus flatteuse.
  2. Le z affiché est brut ; le rapport rappelle explicitement le seuil
     ajusté au nombre de tranches. Une tranche à 2,1 σ parmi 5 n'est pas
     un résultat.

Et le contrôle décisif reste la STABILITÉ : une tranche qui gagne doit gagner
sur la majorité des plis, pas seulement en agrégé.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def test_row_indices(val_end: int, n_predictions: int,
                     sequence_length: int) -> np.ndarray:
    """Indices, dans le DataFrame, correspondant aux prédictions du test.

    La fenêtre i du jeu de test couvre les lignes
        [val_end + i, val_end + i + L)
    et sa cible est celle de la ligne
        val_end + i + L - 1

    C'est cette dernière ligne qui porte le régime de marché pertinent : celui
    observé au moment de la prédiction.
    """
    offset = val_end + sequence_length - 1
    return np.arange(offset, offset + n_predictions)


def _accuracy_stats(predictions: np.ndarray, targets: np.ndarray,
                    raw_moves: np.ndarray, cost_pct: float) -> Dict:
    """Statistiques directionnelles et économiques d'un sous-ensemble."""
    n = len(predictions)
    if n == 0:
        return {"n": 0}

    correct = np.sign(predictions) == np.sign(targets)
    accuracy = float(correct.mean())
    sigma = math.sqrt(0.25 / n)
    z = (accuracy - 0.5) / sigma

    # Mouvement absolu moyen, en pourcentage. `raw_moves` contient les
    # rendements logarithmiques NON normalisés : c'est la seule échelle dans
    # laquelle un calcul de rentabilité a un sens.
    mean_move_pct = float(np.abs(raw_moves).mean() * 100.0)

    # Espérance par trade : on gagne le mouvement quand le signe est bon,
    # on le perd sinon.  EV = p·m − (1−p)·m = (2p − 1)·m
    expected_pct = (2.0 * accuracy - 1.0) * mean_move_pct
    net_pct = expected_pct - cost_pct

    return {
        "n": n,
        "accuracy": accuracy,
        "z": z,
        "mean_move_pct": mean_move_pct,
        "expected_pct": expected_pct,
        "net_pct": net_pct,
        "viable": net_pct > 0,
    }


def analyse_by_quantiles(frame: pd.DataFrame, indices: np.ndarray,
                         predictions: np.ndarray, targets: np.ndarray,
                         regime_column: str, target_column: str,
                         n_buckets: int = 5,
                         cost_pct: float = 0.02) -> List[Dict]:
    """Découpe les prédictions en tranches de régime, par quantiles."""
    regime_values = frame[regime_column].to_numpy()[indices]
    raw_moves = frame[target_column].to_numpy()[indices]

    # Quantiles calculés sur les données de test elles-mêmes. Ce n'est pas une
    # fuite : on ne s'en sert que pour DÉCRIRE, pas pour prédire. En revanche,
    # une utilisation en production devrait figer ces seuils sur le train.
    edges = np.quantile(regime_values, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-12  # inclure la borne supérieure

    rows = []
    for i in range(n_buckets):
        mask = (regime_values >= edges[i]) & (regime_values < edges[i + 1])
        stats = _accuracy_stats(predictions[mask], targets[mask],
                                raw_moves[mask], cost_pct)
        stats["bucket"] = i + 1
        stats["low"] = float(edges[i])
        stats["high"] = float(edges[i + 1])
        rows.append(stats)
    return rows


def analyse_by_confidence(predictions: np.ndarray, targets: np.ndarray,
                          raw_moves: np.ndarray, n_buckets: int = 10,
                          cost_pct: float = 0.02) -> List[Dict]:
    """Découpe les prédictions par CONFIANCE du modèle, pas par régime marché.

    LA QUESTION POSÉE
    -----------------
    Les analyses par régime trient selon l'état du marché. Ici on trie selon
    |prédiction| : le modèle est-il plus fiable quand il s'exprime fortement ?

    C'est le seul mécanisme capable de concentrer un avantage diffus. Si les
    10 % de prédictions les plus affirmatives atteignent 56 % au lieu de 51 %,
    on trade dix fois moins souvent mais chaque trade porte — et l'espérance
    par trade peut passer au-dessus des coûts.

    PIÈGE ÉVITÉ
    -----------
    Le tri se fait sur |prédiction|, une quantité produite par le modèle à
    partir des seules données passées. Aucune information de la cible n'entre
    dans le découpage. Trier sur |cible| serait au contraire une fuite
    grossière — et donnerait 100 % de réussite sur la tranche haute.

    Le résultat est cumulatif : la tranche k regroupe les k·10 % prédictions
    les plus confiantes. C'est ce qui correspond à une règle utilisable
    (« ne trader que si |prédiction| dépasse tel seuil »), contrairement à des
    tranches disjointes.
    """
    confidence = np.abs(predictions)
    order = np.argsort(-confidence)  # du plus confiant au moins confiant

    rows = []
    total = len(predictions)
    for k in range(1, n_buckets + 1):
        take = max(1, int(total * k / n_buckets))
        subset = order[:take]

        stats = _accuracy_stats(predictions[subset], targets[subset],
                               raw_moves[subset], cost_pct)
        stats["bucket"] = k
        stats["pct_kept"] = 100.0 * take / total
        stats["threshold"] = float(confidence[order[take - 1]])
        rows.append(stats)
    return rows


def analyse_by_confidence_marginal(predictions: np.ndarray, targets: np.ndarray,
                                   raw_moves: np.ndarray, n_buckets: int = 10,
                                   cost_pct: float = 0.02) -> List[Dict]:
    """Déciles DISJOINTS de confiance — pour le diagnostic, pas pour la règle.

    POURQUOI CE SECOND DÉCOUPAGE EXISTE
    -----------------------------------
    Les tranches cumulatives sont la bonne forme pour une RÈGLE de trading
    (« ne trader que si |prédiction| > seuil »), mais elles sont trompeuses
    pour le DIAGNOSTIC.

    Raison : si le décile de tête est excellent et tout le reste plat, alors
    le top 20 % est mécaniquement un peu moins bon que le top 10 %, le top
    30 % encore moins, et ainsi de suite. Une monotonie parfaite apparaît
    par pure construction arithmétique, sans qu'il existe le moindre gradient.

    Ce piège a été découvert en testant la fonction sur un scénario synthétique
    à pic isolé : elle le classait « informatif et rentable » à tort.

    Les déciles disjoints n'ont pas ce défaut : un pic isolé se voit comme un
    pic (une valeur haute, neuf valeurs plates), un gradient réel se voit
    comme un gradient.
    """
    confidence = np.abs(predictions)
    order = np.argsort(-confidence)
    total = len(predictions)

    rows = []
    for k in range(n_buckets):
        start = int(total * k / n_buckets)
        end = int(total * (k + 1) / n_buckets)
        subset = order[start:end]
        if len(subset) == 0:
            continue
        stats = _accuracy_stats(predictions[subset], targets[subset],
                               raw_moves[subset], cost_pct)
        stats["decile"] = k + 1  # 1 = le plus confiant
        rows.append(stats)
    return rows


def print_marginal_table(rows: List[Dict]) -> None:
    """Déciles disjoints, du plus confiant au moins confiant."""
    print("\n--- Déciles DISJOINTS de confiance (diagnostic) ---")
    print("  Décile 1 = les 10 % les plus affirmatifs, 10 = les plus timides.")
    print(f"{'décile':>8} {'n':>7} {'direction':>10} {'z':>7}")
    print("-" * 36)
    for row in rows:
        print(f"{row['decile']:>8} {row['n']:>7} {row['accuracy']:>9.2%}"
              f" {row['z']:>7.2f}")
    print("-" * 36)


def print_confidence_table(rows: List[Dict], cost_pct: float) -> None:
    """Tableau de l'analyse par confiance, en cumulé."""
    print("\n--- Découpage par CONFIANCE du modèle (cumulé) ---")
    print("  Chaque ligne = les X % de prédictions les plus affirmatives.")
    print(f"{'% gardé':>9} {'seuil':>9} {'n':>7} {'direction':>10} {'z':>7}"
          f" {'mouv.moy':>10} {'espérance':>10} {'net':>9}")
    print("-" * 80)
    for row in rows:
        if row["n"] == 0:
            continue
        flag = " *" if row["viable"] else ""
        print(f"{row['pct_kept']:>8.0f}% {row['threshold']:>9.3f}"
              f" {row['n']:>7} {row['accuracy']:>9.2%} {row['z']:>7.2f}"
              f" {row['mean_move_pct']:>9.3f}% {row['expected_pct']:>9.3f}%"
              f" {row['net_pct']:>+8.3f}%{flag}")
    print("-" * 80)
    print(f"  * = espérance nette positive après {cost_pct:.3f} % de coûts")
    print("  Les tranches étant cumulatives et emboîtées, elles ne sont pas")
    print("  indépendantes : la correction de Bonferroni y serait trop")
    print("  sévère. Ce qui compte ici est la MONOTONIE — la direction doit")
    print("  croître régulièrement quand on resserre le filtre. Un pic isolé")
    print("  sur une seule tranche est du bruit.")


def monotonicity_score(rows: List[Dict]) -> Dict:
    """Existe-t-il un GRADIENT de précision selon la confiance ?

    Mesuré sur les déciles DISJOINTS (voir analyse_by_confidence_marginal).
    On attend `rows` produit par cette fonction, pas par la version cumulative.

    Deux indicateurs :
      - corrélation de rang entre décile et précision : un gradient réel la
        rend fortement positive
      - `spike_ratio` : de combien le décile de tête dépasse la moyenne des
        autres, en écarts-types. Un pic isolé donne un ratio élevé avec une
        corrélation médiocre.
    """
    valid = [r for r in rows if r["n"] >= 50]
    if len(valid) < 3:
        return {"n": 0}

    # Ordre : du moins confiant (décile 10) au plus confiant (décile 1), pour
    # qu'une corrélation POSITIVE signifie « plus confiant = plus précis ».
    ordered = sorted(valid, key=lambda r: -r["decile"])
    accuracies = np.array([r["accuracy"] for r in ordered])

    # Corrélation de rang entre « resserrement » et précision.
    ranks = np.arange(len(accuracies))
    if accuracies.std() == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(ranks, accuracies)[0, 1])

    increases = int((np.diff(accuracies) > 0).sum())

    # Détection de pic : le décile de tête (dernier après tri) se détache-t-il
    # anormalement du reste ?
    top = accuracies[-1]
    others = accuracies[:-1]
    spread = others.std(ddof=1) if len(others) > 1 else 0.0
    spike_ratio = (top - others.mean()) / spread if spread > 0 else 0.0

    return {
        "n": len(ordered),
        "correlation": correlation,
        "increases": increases,
        "steps": len(accuracies) - 1,
        "widest": float(accuracies[0]),    # décile le moins confiant
        "tightest": float(top),            # décile le plus confiant
        "spike_ratio": float(spike_ratio),
    }


def print_confidence_verdict(rows: List[Dict], marginal_rows: List[Dict],
                             cost_pct: float) -> None:
    """Verdict de l'analyse par confiance.

    `rows`          : tranches cumulatives (pour la règle de trading)
    `marginal_rows` : déciles disjoints (pour juger du gradient)
    """
    mono = monotonicity_score(marginal_rows)
    print("\n" + "=" * 80)
    if mono["n"] == 0:
        print("Trop peu de données pour juger.")
        print("=" * 80)
        return

    print(f"  Décile le MOINS confiant       : {mono['widest']:.2%}")
    print(f"  Décile le PLUS confiant        : {mono['tightest']:.2%}")
    print(f"  Gradient                       : {mono['increases']}/{mono['steps']} "
          f"étapes en hausse (corrélation {mono['correlation']:+.2f})")
    print(f"  Indice de pic isolé            : {mono['spike_ratio']:.2f} σ")

    viable = [r for r in rows if r.get("viable") and r["n"] >= 100]
    # Gradient réel : corrélation forte sur les déciles DISJOINTS...
    strong_mono = mono["correlation"] > 0.6 and \
        mono["increases"] >= mono["steps"] * 0.6
    # ...et pas un simple pic sur le décile de tête. Au-delà de 3 σ, le décile
    # de tête se détache trop du reste pour qu'on parle de gradient.
    is_spike = mono["spike_ratio"] > 3.0
    strong_mono = strong_mono and not is_spike

    print("-" * 80)
    if viable and strong_mono:
        best = max(viable, key=lambda r: r["net_pct"])
        print("VERDICT : la confiance du modèle est informative ET rentable.")
        print(f"  Meilleur point : garder {best['pct_kept']:.0f} % des signaux")
        print(f"  -> {best['n']} trades, direction {best['accuracy']:.2%}, "
              f"net {best['net_pct']:+.3f} % par trade.")
        print("\n  Prochaine étape OBLIGATOIRE avant toute conclusion : un")
        print("  backtest avec slippage, exécution réaliste et gestion du")
        print("  risque. Une espérance positive par trade n'est pas une")
        print("  stratégie — il reste à vérifier la trajectoire du capital.")
    elif strong_mono:
        print("VERDICT : la confiance est informative, mais insuffisante.")
        print("  La précision progresse bien quand on resserre le filtre, ce")
        print("  qui valide le mécanisme — mais aucune tranche ne couvre les")
        print(f"  coûts de {cost_pct:.3f} %. Un courtier moins cher, ou un")
        print("  timeframe à mouvements plus amples, pourrait changer cela.")
    elif viable and is_spike:
        print("VERDICT : pic isolé, très probablement un artefact.")
        print(f"  Le décile le plus confiant dépasse les autres de "
              f"{mono['spike_ratio']:.1f} écarts-types,")
        print("  sans gradient régulier derrière. Un effet réel de confiance")
        print("  produit une progression continue, pas une marche d'escalier.")
        print("\n  À vérifier avant tout enthousiasme : ce décile est-il")
        print("  concentré sur une seule période ? Relancez avec --regime et")
        print("  regardez la stabilité entre plis.")
    elif viable:
        print("VERDICT : rentabilité apparente, sans gradient.")
        print("  Une tranche passe le seuil de rentabilité, mais la précision")
        print("  ne progresse pas avec la confiance. Profil d'artefact.")
    else:
        print("VERDICT : la confiance du modèle n'est pas informative.")
        print("  Ses prédictions fortes ne sont pas plus fiables que ses")
        print("  prédictions timides. C'est cohérent avec un modèle qui")
        print("  capte un signal faible et uniformément réparti.")
        print("\n  Vous avez fait le tour honnête de la question : ce jeu de")
        print("  features ne permet pas de trading directionnel rentable.")
    print("=" * 80)


def print_regime_table(rows: List[Dict], regime_column: str,
                       cost_pct: float) -> None:
    """Tableau d'une analyse de régime."""
    n_buckets = len([r for r in rows if r["n"] > 0])
    print(f"\n--- Découpage par {regime_column} ({n_buckets} tranches) ---")
    print(f"{'tranche':>8} {'plage':>18} {'n':>7} {'direction':>10} {'z':>7}"
          f" {'mouv.moy':>10} {'espérance':>10} {'net':>9}")
    print("-" * 84)
    for row in rows:
        if row["n"] == 0:
            continue
        flag = " *" if row["viable"] else ""
        plage = f"{row['low']:.4f}–{row['high']:.4f}"
        print(f"{row['bucket']:>8} {plage:>18} {row['n']:>7}"
              f" {row['accuracy']:>9.2%} {row['z']:>7.2f}"
              f" {row['mean_move_pct']:>9.3f}% {row['expected_pct']:>9.3f}%"
              f" {row['net_pct']:>+8.3f}%{flag}")
    print("-" * 84)
    print(f"  * = espérance nette positive après {cost_pct:.3f} % de coûts")

    # Seuil ajusté aux comparaisons multiples : approximation de Bonferroni
    # sur le quantile de la loi normale.
    if n_buckets > 1:
        adjusted = _bonferroni_z(n_buckets)
        print(f"  Avec {n_buckets} tranches, le seuil de significativité passe")
        print(f"  de 1,96 à environ {adjusted:.2f}. Une tranche en dessous de")
        print("  ce z n'est pas un résultat, c'est une tranche chanceuse.")


def _bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    """Quantile normal pour alpha/n, par recherche binaire sur erf."""
    target = 1.0 - (alpha / n_tests) / 2.0
    low, high = 0.0, 6.0
    for _ in range(80):
        mid = (low + high) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def stability_across_folds(per_fold: Dict[int, List[Dict]],
                           bucket: int) -> Dict:
    """Une tranche donnée gagne-t-elle sur la MAJORITÉ des plis ?

    C'est le contrôle qui distingue un régime réellement favorable d'une
    tranche qui doit tout à un ou deux plis exceptionnels.
    """
    accuracies = []
    for rows in per_fold.values():
        match = [r for r in rows if r["bucket"] == bucket and r["n"] > 30]
        if match:
            accuracies.append(match[0]["accuracy"])

    if not accuracies:
        return {"n_folds": 0}

    values = np.array(accuracies)
    return {
        "n_folds": len(values),
        "above_half": int((values > 0.5).sum()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def print_stability(per_fold: Dict[int, List[Dict]], n_buckets: int) -> None:
    """Stabilité de chaque tranche à travers les plis."""
    print("\n--- Stabilité par tranche, à travers les plis ---")
    print(f"{'tranche':>8} {'plis':>6} {'>50%':>6} {'moyenne':>10}"
          f" {'écart-type':>11} {'min':>8} {'max':>8}")
    print("-" * 62)
    for bucket in range(1, n_buckets + 1):
        stats = stability_across_folds(per_fold, bucket)
        if stats["n_folds"] == 0:
            continue
        print(f"{bucket:>8} {stats['n_folds']:>6}"
              f" {stats['above_half']:>3}/{stats['n_folds']:<2}"
              f" {stats['mean']:>9.2%} {stats['std']:>10.2%}"
              f" {stats['min']:>7.2%} {stats['max']:>7.2%}")
    print("-" * 62)
    print("  Une tranche exploitable est au-dessus de 50 % sur la plupart des")
    print("  plis, avec un écart-type modeste. Une moyenne élevée portée par")
    print("  un seul pli ne vaut rien : vous ne pouvez pas savoir à l'avance")
    print("  si vous êtes dans ce pli-là.")


def final_verdict(rows: List[Dict], per_fold: Dict[int, List[Dict]],
                  n_buckets: int) -> None:
    """Synthèse : existe-t-il un régime exploitable ?"""
    print("\n" + "=" * 84)
    threshold = _bonferroni_z(n_buckets)

    candidates = []
    for row in rows:
        if row["n"] == 0:
            continue
        if row["z"] < threshold or not row["viable"]:
            continue
        stability = stability_across_folds(per_fold, row["bucket"])
        if stability["n_folds"] >= 4 and \
                stability["above_half"] >= stability["n_folds"] * 0.7:
            candidates.append((row, stability))

    if candidates:
        print("VERDICT : au moins un régime paraît exploitable.")
        for row, stability in candidates:
            print(f"  Tranche {row['bucket']} — direction {row['accuracy']:.2%}, "
                  f"net {row['net_pct']:+.3f} %, "
                  f"gagnante sur {stability['above_half']}/{stability['n_folds']} plis.")
        print("\n  Prochaine étape : re-mesurer en n'entraînant QUE sur ce")
        print("  régime, puis backtester avec slippage et frais réels.")
    else:
        print("VERDICT : aucun régime ne franchit les trois critères")
        print("(significativité ajustée, rentabilité après coûts, stabilité")
        print("entre plis).")
        print("\n  Ce n'est pas un échec de méthode : c'est l'information que")
        print("  le filtrage par régime ne sauve pas cette configuration.")
        print("  Le signal directionnel existe, mais il reste sous les coûts.")
    print("=" * 84)