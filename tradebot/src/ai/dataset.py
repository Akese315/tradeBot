"""
Datasets PyTorch pour séries temporelles (remplace `LSTM_Dataset`,
`BasicDataset`, `GeneralDataset` et la fonction `create_batches` qui était
dupliquée à l'identique dans `model.py` ET `AI.py`).

DEUX RÈGLES SPÉCIFIQUES AUX SÉRIES TEMPORELLES
----------------------------------------------
1. **Découpage chronologique, jamais aléatoire.** Train = passé,
   validation = milieu, test = futur. Un split aléatoire mettrait des bougies
   de 2023 dans le train et des bougies de 2022 dans le test : le modèle
   « connaîtrait le futur ». (L'ancien découpage était déjà chronologique —
   c'était correct, on le conserve.)

2. **Le scaler s'ajuste UNIQUEMENT sur le train.** Calculer moyenne et
   écart-type sur l'ensemble des données fait fuiter des statistiques du
   futur dans le train. C'est subtil et c'est la fuite la plus fréquente
   dans les projets de trading ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.ai.features import (FEATURE_COLUMNS, MIN_VOLATILITY, TARGET_COLUMN,
                             VOLATILITY_COLUMN)


class SequenceDataset(Dataset):
    """Fenêtres glissantes pour LSTM.

    Chaque échantillon i vaut :
        X[i] = features[i : i + L]   -> tenseur (L, nb_features)
        y[i] = target[i + L - 1]     -> scalaire

    ⚠️ Correction majeure : l'ancien `AI.trainModel` faisait
    `inputs.unsqueeze(1)`, ce qui produisait une séquence de LONGUEUR 1.
    Le LSTM ne voyait donc qu'une seule bougie à la fois — autant utiliser un
    simple perceptron : toute la mémoire temporelle, la seule raison d'être
    d'un LSTM, était désactivée. Ici la dimension temporelle est réelle.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 sequence_length: int) -> None:
        self.sequence_length = sequence_length
        # `float32` : les GPU sont ~2x plus rapides qu'en float64, et la
        # précision supplémentaire n'apporte rien sur des rendements.
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return max(0, len(self.features) - self.sequence_length + 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.features[index: index + self.sequence_length]
        target = self.targets[index + self.sequence_length - 1]
        return window, target


@dataclass
class SplitData:
    """Conteneur des trois sous-ensembles + le scaler ajusté sur le train."""
    train: SequenceDataset
    validation: SequenceDataset
    test: SequenceDataset
    scaler: StandardScaler
    target_scale: float  # écart-type de la cible, pour dénormaliser ensuite


def build_splits(frame: pd.DataFrame, sequence_length: int,
                 train_ratio: float, val_ratio: float,
                 rolling_target_scale: bool = True) -> SplitData:
    """Découpe le DataFrame en train/val/test et normalise proprement.

    `rolling_target_scale` :
        True  -> chaque cible est divisée par la volatilité réalisée À CET
                 INSTANT (fenêtre de 20 bougies passées). La cible devient un
                 mouvement EXPRIMÉ EN ÉCARTS-TYPES LOCAUX, comparable entre
                 2010 et 2023 malgré le changement de régime.
        False -> ancien comportement : une constante unique calculée sur le
                 train. À conserver pour reproduire les runs antérieurs.
    """
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)

    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # --- Normalisation des entrées -------------------------------------
    scaler = StandardScaler()
    scaler.fit(features[:train_end])          # AJUSTEMENT SUR LE TRAIN SEUL
    features = scaler.transform(features)     # application partout

    # --- Normalisation de la cible -------------------------------------
    # Les rendements horaires valent ~0,001. Sans mise à l'échelle, les
    # gradients sont minuscules et l'apprentissage traîne.
    if rolling_target_scale and VOLATILITY_COLUMN in frame.columns:
        # NORMALISATION GLISSANTE.
        # On divise chaque cible par la volatilité locale, mesurée sur les 20
        # bougies PRÉCÉDENTES. Un mouvement de 1 % dans un marché calme et un
        # mouvement de 3 % dans un marché agité deviennent la même valeur : ce
        # que le modèle apprend est « ce mouvement est-il grand POUR L'ÉPOQUE »,
        # une question qui garde le même sens en 2010 et en 2023.
        #
        # Aucune fuite : la volatilité au temps t n'utilise que t-20..t, alors
        # que la cible porte sur t+1.
        local_vol = frame[VOLATILITY_COLUMN].to_numpy(dtype=np.float64)
        local_vol = np.maximum(local_vol, MIN_VOLATILITY)
        targets = targets / local_vol
        # Conservé pour le journal : ordre de grandeur moyen du diviseur.
        target_scale = float(np.mean(local_vol[:train_end]))
    else:
        # ANCIEN COMPORTEMENT : une constante unique, calculée sur le train.
        target_scale = float(np.std(targets[:train_end])) or 1.0
        targets = targets / target_scale

    def slice_dataset(start: int, end: int) -> SequenceDataset:
        return SequenceDataset(features[start:end], targets[start:end],
                               sequence_length)

    return SplitData(
        train=slice_dataset(0, train_end),
        validation=slice_dataset(train_end, val_end),
        test=slice_dataset(val_end, n),
        scaler=scaler,
        target_scale=target_scale,
    )