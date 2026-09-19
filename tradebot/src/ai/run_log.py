"""
Journalisation des entraînements.

Chaque exécution ajoute un bloc à `historique_runs.txt` à la racine du projet.
Le fichier n'est JAMAIS écrasé (mode "a" = append) : vous accumulez l'historique
complet de vos expériences.

POURQUOI C'EST IMPORTANT
------------------------
Sans journal, au bout de vingt essais vous ne savez plus quelle combinaison de
paramètres a donné quel résultat. Vous refaites deux fois la même expérience,
vous croyez qu'un changement a aidé alors que c'était le run précédent, et vous
ne pouvez rien reproduire. C'est exactement ce qui est arrivé à vos anciennes
« version 13, 14, 15, 16, 17 » : le README note les tailles de couche cachée,
mais aucune métrique, aucune date, aucun jeu de données associé.

Un journal en texte brut est volontairement primitif : lisible sans outil,
greppable, diffable, et il survivra à ce projet.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOG_FILENAME = "historique_runs.txt"


def _format_block(title: str, values: Dict[str, Any], width: int = 34) -> str:
    """Formate un bloc `clé ......... valeur` aligné."""
    lines = [f"  [{title}]"]
    for key, value in values.items():
        if isinstance(value, float):
            # 6 décimales : assez pour distinguer deux losses proches,
            # sans noyer la lecture.
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        lines.append(f"    {key:<{width}} {rendered}")
    return "\n".join(lines)


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Convertit une dataclass de configuration en dictionnaire."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"valeur": str(obj)}


def append_run(report: Dict[str, Any],
               model_config: Any,
               indicator_config: Any = None,
               context: Optional[Dict[str, Any]] = None,
               log_path: Optional[Path] = None) -> Path:
    """Ajoute un bloc au journal et renvoie le chemin du fichier.

    `report`           : dictionnaire renvoyé par `train()`
    `model_config`     : instance de ModelConfig
    `indicator_config` : instance de IndicatorConfig (optionnel)
    `context`          : infos libres (symbole, période, nb de bougies…)
    """
    if log_path is None:
        # Racine du projet = deux niveaux au-dessus de src/ai/
        log_path = Path(__file__).resolve().parent.parent.parent / LOG_FILENAME

    history = report.get("history", [])
    best_epoch = None
    if history:
        best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]

    sections = [
        "=" * 78,
        f"RUN : {report.get('run_name', 'sans_nom')}",
        f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 78,
    ]

    if context:
        sections.append(_format_block("DONNÉES", context))

    sections.append(_format_block("MODÈLE", _as_dict(model_config)))

    if indicator_config is not None:
        sections.append(_format_block("INDICATEURS", _as_dict(indicator_config)))

    results = {
        "epoques_effectuees": len(history),
        "meilleure_epoque": best_epoch,
        "meilleure_loss_val": report.get("best_val_loss"),
        "loss_test": report.get("test_loss"),
        "loss_baseline_zero": report.get("baseline_loss_zero_prediction"),
        "bat_la_baseline": "OUI" if report.get("beats_baseline") else "NON",
        "precision_directionnelle": f"{report.get('test_direction_accuracy', 0):.2%}",
    }
    sections.append(_format_block("RÉSULTATS", results))

    # Courbe d'apprentissage condensée : permet de repérer d'un coup d'œil
    # à quelle époque la validation a décroché (= début du sur-apprentissage).
    if history:
        sections.append("  [COURBE]")
        sections.append(f"    {'époque':>8} {'train':>10} {'val':>10} {'direction':>10}")
        for entry in history:
            marker = " <-- meilleur" if entry["epoch"] == best_epoch else ""
            sections.append(
                f"    {entry['epoch']:>8} {entry['train_loss']:>10.5f} "
                f"{entry['val_loss']:>10.5f} {entry['val_direction']:>9.1%}{marker}"
            )

    sections.append("")  # ligne vide entre deux runs

    block = "\n".join(sections) + "\n"

    # `encoding="utf-8"` explicite : sans lui, Windows écrit en cp1252 et
    # les accents deviennent illisibles.
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(block)

    return log_path
