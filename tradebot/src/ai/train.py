"""
Boucle d'entraînement et d'évaluation (remplace `AI.py`).

CE QUI CHANGE FONDAMENTALEMENT PAR RAPPORT À L'ANCIENNE VERSION
---------------------------------------------------------------
1. **Early stopping sur la validation.** L'ancien code faisait 15 époques en
   aveugle, ne regardait jamais la loss de validation, puis évaluait le
   modèle sur le test... en rechargeant `version_1.pth` depuis le disque,
   c'est-à-dire potentiellement un modèle d'une session précédente.

2. **Sauvegarde du MEILLEUR modèle, pas du dernier.** On conserve les poids
   de l'époque où la validation était minimale.

3. **Métrique honnête : la précision directionnelle.** En trading, une MSE
   basse ne rapporte rien. Ce qui compte est : « le modèle prévoit-il le bon
   SIGNE ? » 50 % = pile ou face. 53–55 % de façon stable est déjà un
   résultat sérieux. Au-delà de 60 % sur des données horaires, chercher la
   fuite de données avant de se réjouir.

4. **Comparaison à une baseline naïve.** On mesure systématiquement le modèle
   contre « prédire 0 » (aucun mouvement). Si le réseau ne bat pas ça, il
   n'apporte rien — et c'était très probablement le cas de la version
   précédente sans qu'aucune métrique ne le révèle.

5. **`input()` supprimé de la fonction.** Demander « voulez-vous sauvegarder ? »
   au milieu d'une routine bloque toute exécution automatisée (cron, CI,
   notebook). La décision revient à l'appelant, via un argument.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import MODELS_DIR, ModelConfig
from src.ai.dataset import SplitData
from src.ai.features import FEATURE_COLUMNS
from src.ai.model import TradingModel
from src.ai.run_log import append_run


def _build_loss(name: str) -> nn.Module:
    """Fabrique la fonction de perte à partir de son nom.

    ATTENTION : deux runs utilisant des pertes différentes ne sont PAS
    comparables entre eux — les échelles diffèrent. Comparez toujours
    modèle contre baseline à l'intérieur du même run.
    """
    losses = {
        "huber": nn.SmoothL1Loss,   # robuste aux valeurs extrêmes
        "mse": nn.MSELoss,          # pénalise fort les grosses erreurs
        "l1": nn.L1Loss,            # très robuste, convergence plus lente
    }
    if name not in losses:
        raise ValueError(f"Perte inconnue : {name}. Choix : {list(losses)}")
    return losses[name]()


def get_device() -> torch.device:
    """CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
               device: torch.device, optimizer: torch.optim.Optimizer | None
               ) -> Tuple[float, float]:
    """Exécute une époque. Si `optimizer` est None, on est en évaluation.

    Renvoie (loss moyenne, précision directionnelle).
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, total_samples, correct_direction = 0.0, 0, 0

    # `no_grad` en éval : divise par ~2 la mémoire et accélère nettement.
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                # Clipping du gradient : sans lui, les LSTM subissent des
                # « exploding gradients » qui envoient la loss à NaN.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            correct_direction += (
                torch.sign(predictions) == torch.sign(targets)
            ).sum().item()

    return total_loss / total_samples, correct_direction / total_samples


def train(splits: SplitData, config: ModelConfig | None = None,
          run_name: str | None = None, save: bool = True,
          indicator_config=None, context: Dict | None = None) -> Dict:
    """Entraîne le modèle et renvoie un rapport de métriques.

    `indicator_config` et `context` ne servent qu'à enrichir le journal
    `historique_runs.txt` : ils n'influencent pas l'entraînement.
    """
    config = config or ModelConfig()
    device = get_device()
    print(f"Périphérique utilisé : {device}")

    # `shuffle=True` sur le TRAIN uniquement : mélanger les FENÊTRES est sain
    # (chaque fenêtre est déjà un échantillon complet et ordonné en interne),
    # cela décorrèle les batches. En revanche on ne mélange jamais val/test,
    # pour pouvoir tracer les prédictions dans l'ordre chronologique.
    train_loader = DataLoader(splits.train, batch_size=config.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(splits.validation, batch_size=config.batch_size)
    test_loader = DataLoader(splits.test, batch_size=config.batch_size)

    model = TradingModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # Huber (SmoothL1) par défaut plutôt que MSE : les marchés produisent des
    # valeurs extrêmes (résultats trimestriels, krachs). La MSE élève l'erreur
    # au carré, donc quelques bougies aberrantes dominent tout le gradient.
    loss_fn = _build_loss(config.loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    best_val_loss = float("inf")
    best_state = None
    epochs_without_progress = 0
    history = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, loss_fn,
                                           device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, loss_fn, device, None)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_direction": val_acc})
        print(f"Époque {epoch:3d} | train {train_loss:.5f} "
              f"| val {val_loss:.5f} | direction val {val_acc:.1%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # `.cpu()` avant la copie : évite de saturer la VRAM du GPU.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
            if epochs_without_progress >= config.patience:
                print(f"Arrêt anticipé : {config.patience} époques sans progrès.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Évaluation finale sur le TEST, jamais vu pendant l'entraînement.
    # ------------------------------------------------------------------
    test_loss, test_acc = _run_epoch(model, test_loader, loss_fn, device, None)
    baseline = _naive_baseline(test_loader, loss_fn, device)

    report = {
        "run_name": run_name,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_direction_accuracy": test_acc,
        "baseline_loss_zero_prediction": baseline,
        "beats_baseline": test_loss < baseline,
        "history": history,
    }

    print("\n===== RAPPORT =====")
    print(f"Loss test           : {test_loss:.5f}")
    print(f"Loss baseline (0)   : {baseline:.5f}")
    print(f"Précision direction : {test_acc:.2%}  (50 % = hasard)")
    if not report["beats_baseline"]:
        print("⚠️ Le modèle ne bat PAS la prédiction nulle : il n'apporte "
              "aucune information exploitable en l'état.")

    # Journalisation : TOUJOURS, même si save=False. Un essai raté est une
    # information autant qu'un essai réussi — c'est même le plus fréquent.
    log_path = append_run(report, config, indicator_config, context)
    print(f"Run consigné dans {log_path}")

    if save:
        _persist(model, splits, report, run_name)
    return report


def _naive_baseline(loader: DataLoader, loss_fn: nn.Module,
                    device: torch.device) -> float:
    """Loss obtenue en prédisant systématiquement 0 (aucun mouvement).
    C'est le seuil minimal que tout modèle doit battre."""
    total, count = 0.0, 0
    with torch.no_grad():
        for _, targets in loader:
            targets = targets.to(device)
            zeros = torch.zeros_like(targets)
            total += loss_fn(zeros, targets).item() * targets.size(0)
            count += targets.size(0)
    return total / count


def _persist(model: nn.Module, splits: SplitData, report: Dict,
             run_name: str) -> None:
    """Sauvegarde poids + scaler + métriques dans un dossier daté.

    ⚠️ Le SCALER doit impérativement être sauvegardé avec le modèle. Sans lui,
    impossible de normaliser les données en production de la même façon qu'à
    l'entraînement — le modèle recevrait des entrées d'une autre échelle et
    produirait n'importe quoi. C'était absent de l'ancien code, qui écrasait
    par ailleurs toujours le même fichier `version_1.pth`.
    """
    run_dir: Path = MODELS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pth")
    joblib.dump(splits.scaler, run_dir / "scaler.pkl")
    (run_dir / "metrics.json").write_text(
        json.dumps({**report, "target_scale": splits.target_scale}, indent=2)
    )
    print(f"Modèle sauvegardé dans {run_dir}")