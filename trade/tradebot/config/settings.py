"""
Configuration centralisée du projet.

RÈGLE D'OR : aucun secret, aucun chemin, aucune période magique ne doit être
écrit en dur ailleurs que dans ce fichier.

⚠️ Faille corrigée : l'ancien `bot.py` contenait la clé API Finnhub ET le mot
de passe MySQL en clair dans le code source. Si ce dépôt a déjà été poussé sur
GitHub, ces identifiants sont compromis — il faut les RÉVOQUER et en générer
de nouveaux, même si le dépôt est privé (l'historique Git garde tout).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Charge le fichier .env situé à la racine du projet.
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# ==========================================================================
# CHEMINS
# ==========================================================================
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # CSV bruts téléchargés (ex-"dataframe/")
FEATURES_DIR = DATA_DIR / "features"  # CSV enrichis (ex-"analysed_dataframe/")
MODELS_DIR = ROOT_DIR / "models"      # poids .pth + scalers .pkl

for _d in (RAW_DIR, FEATURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# SECRETS (lus depuis .env — jamais commités)
# ==========================================================================
@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "")
    database: str = os.getenv("DB_NAME", "TRADEBOT")


ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")


# ==========================================================================
# PÉRIODES DES INDICATEURS
# ==========================================================================
# ⚠️ Point conceptuel majeur : ces périodes sont en NOMBRE DE BOUGIES.
# Les données sont en H1 et le NASDAQ est ouvert 6,5 h/jour ≈ 7 bougies/jour.
# L'ancien code utilisait `NASDAQ_HOUR_A_DAY = 6`, puis appelait ailleurs
# `calculate_Standard(16)` : on mélangeait donc des SMA « 20 jours » (120
# bougies) et des SMA « 16 bougies » (~2 jours) selon le point d'entrée.
# Les datasets produits n'étaient pas comparables entre eux.
BARS_PER_DAY = 23 #7 si c'était stocks, 23 sinon

@dataclass(frozen=True)
class IndicatorConfig:
    sma_period: int = 20 * BARS_PER_DAY      # tendance moyen terme (~20 jours)
    stoch_k_period: int = 14 * BARS_PER_DAY  # standard Wilder : 14 périodes
    stoch_d_period: int = 3 * BARS_PER_DAY   # ligne de signal : 3 périodes
    rsi_period: int = 14 * BARS_PER_DAY
    adx_period: int = 14 * BARS_PER_DAY


# ==========================================================================
# HYPERPARAMÈTRES DU MODÈLE
# ==========================================================================
@dataclass(frozen=True)
class ModelConfig:
    sequence_length: int = 48    # nb de bougies vues par le LSTM (≈ 1 semaine)
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 50
    patience: int = 7            # early stopping : nb d'époques sans progrès
    train_ratio: float = 0.70
    val_ratio: float = 0.15      # le reste (15 %) sert de test final

    # --- Régularisation et perte -----------------------------------------
    # Ces trois réglages vivaient auparavant en dur dans train.py et
    # dataset.py, donc impossibles à balayer. Les remonter ici les rend
    # accessibles à sweep.py comme n'importe quel autre hyperparamètre.
    weight_decay: float = 1e-4   # régularisation L2 de l'optimiseur
    loss_name: str = "huber"     # "huber" | "mse" | "l1"
    rolling_target_scale: bool = True  # normalisation glissante de la cible