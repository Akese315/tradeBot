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
    feature_columns: list  # colonnes réellement utilisées, dans l'ordre

    @property
    def n_features(self) -> int:
        """Nombre d'entrées du modèle. À lire ici plutôt que sur la constante
        FEATURE_COLUMNS : selon l'actif, le volume peut être absent."""
        return len(self.feature_columns)


def build_splits_from_bounds(frame: pd.DataFrame, sequence_length: int,
                             train_start: int, train_end: int,
                             val_end: int, test_end: int,
                             rolling_target_scale: bool = True,
                             feature_columns: list | None = None) -> SplitData:
    """Version à bornes explicites, utilisée par le walk-forward.

    Les quatre bornes définissent trois blocs strictement chronologiques :
        train = [train_start, train_end)
        val   = [train_end,   val_end)
        test  = [val_end,     test_end)

    Le scaler est ajusté sur le train de CE pli uniquement. C'est le point
    critique du walk-forward : réutiliser un scaler global ferait entrer des
    statistiques du futur dans chaque pli, et la validation perdrait tout sens.
    """
    feature_columns = feature_columns or list(FEATURE_COLUMNS)
    features = frame[feature_columns].to_numpy(dtype=np.float64)
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    scaler.fit(features[train_start:train_end])
    features = scaler.transform(features)

    if rolling_target_scale and VOLATILITY_COLUMN in frame.columns:
        local_vol = np.maximum(
            frame[VOLATILITY_COLUMN].to_numpy(dtype=np.float64), MIN_VOLATILITY)
        targets = targets / local_vol
        target_scale = float(np.mean(local_vol[train_start:train_end]))
    else:
        target_scale = float(np.std(targets[train_start:train_end])) or 1.0
        targets = targets / target_scale

    def slice_dataset(start: int, end: int) -> SequenceDataset:
        return SequenceDataset(features[start:end], targets[start:end],
                               sequence_length)

    return SplitData(
        train=slice_dataset(train_start, train_end),
        validation=slice_dataset(train_end, val_end),
        test=slice_dataset(val_end, test_end),
        scaler=scaler,
        target_scale=target_scale,
        feature_columns=feature_columns,
    )


def build_splits(frame: pd.DataFrame, sequence_length: int,
                 train_ratio: float, val_ratio: float,
                 rolling_target_scale: bool = True,
                 feature_columns: list | None = None) -> SplitData:
    """Découpe le DataFrame en train/val/test et normalise proprement.

    `rolling_target_scale` :
        True  -> chaque cible est divisée par la volatilité réalisée À CET
                 INSTANT (fenêtre de 20 bougies passées). La cible devient un
                 mouvement EXPRIMÉ EN ÉCARTS-TYPES LOCAUX, comparable entre
                 2010 et 2023 malgré le changement de régime.
        False -> ancien comportement : une constante unique calculée sur le
                 train. À conserver pour reproduire les runs antérieurs.
    """
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return build_splits_from_bounds(
        frame, sequence_length, 0, train_end, val_end, n,
        rolling_target_scale=rolling_target_scale,
        feature_columns=feature_columns)