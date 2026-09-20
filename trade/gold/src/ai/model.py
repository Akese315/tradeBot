"""
Définition du réseau de neurones (anciennement `model.py`, mêlé aux datasets).

⚠️ Bug bloquant corrigé : `AI.py` appelait `tradingModel()` (minuscule, sans
argument) alors que la classe s'appelait `TradingModel` et exigeait 4
arguments — et `AI.py` n'importait même pas `model.py`. Le script
d'entraînement ne pouvait donc pas s'exécuter du tout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TradingModel(nn.Module):
    """LSTM + tête linéaire, pour la régression d'un rendement futur.

    Choix d'architecture et justification :

    - **LSTM plutôt que GRU/Transformer** : sur quelques dizaines de milliers
      de bougies, un Transformer sur-apprend immédiatement. Le LSTM reste le
      bon compromis à cette taille de données.

    - **`batch_first=True`** : les tenseurs sont (batch, temps, features),
      l'ordre le plus lisible. Attention, ce n'est PAS le défaut de PyTorch.

    - **On ne garde que le dernier pas de temps** (`output[:, -1, :]`) : c'est
      l'état caché qui a « vu » toute la séquence. Correct dans l'ancien code,
      conservé ici.

    - **`num_layers = 2` au lieu de 3** : avec 3 couches et 600 unités
      cachées, le modèle comptait ~10 M de paramètres pour quelques dizaines
      de milliers d'échantillons bruités. Ratio absurde — le réseau
      mémorisait le train. Moins de capacité = meilleure généralisation ici.

    - **Sortie à 1 valeur** : on prédit le rendement du close suivant.
      L'ancienne version prédisait simultanément (close, high, low) sans
      pondérer les trois pertes ; en pratique le modèle optimisait surtout
      la composante la plus grande. Un objectif = un modèle plus lisible.
    """

    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # PyTorch ignore `dropout` si num_layers == 1 (et émet un warning).
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # LayerNorm sur l'état caché : stabilise nettement l'entraînement des
        # LSTM profonds, à coût quasi nul.
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, sequence_length, input_size) -> (batch,)

        Note : inutile d'initialiser h0/c0 à la main comme le faisait
        l'ancien code — PyTorch le fait déjà avec des zéros, sur le bon
        device. C'était du code correct mais redondant, et une source
        d'erreur `device mismatch` si on oublie le `.to(x.device)`.
        """
        output, _ = self.lstm(x)
        last_hidden = self.norm(output[:, -1, :])
        return self.head(self.dropout(last_hidden)).squeeze(-1)
