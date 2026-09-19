"""
Chargeur CSV universel.

POURQUOI CE FICHIER EXISTE
--------------------------
Vos CSV d'actions viennent d'AlphaVantage et ont un format connu :
`time, symbol, highPrice, lowPrice, closePrice, openPrice, volume`.

Les jeux de données d'or n'ont aucune convention commune. On rencontre :
    Date;Time;Open;High;Low;Close;Volume         (MetaTrader, séparateur ;)
    Datetime,Open,High,Low,Close,Volume          (Kaggle)
    Gmt time,Open,High,Low,Close,Volume          (Dukascopy)
    time,open,high,low,close                     (sans volume du tout)
    timestamp,o,h,l,c,v                          (formats API abrégés)

Écrire un chargeur par source serait interminable. Celui-ci détecte les
colonnes et le format de date automatiquement, et échoue avec un message
clair quand il n'y arrive pas — plutôt que de produire silencieusement des
données décalées.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.entities import Quote

# --------------------------------------------------------------------------
# Alias connus pour chaque colonne, en minuscules et sans espaces.
# L'ordre compte : le premier trouvé gagne.
# --------------------------------------------------------------------------
ALIASES: dict[str, List[str]] = {
    "open":   ["open", "openprice", "o", "open_price", "ouverture"],
    "high":   ["high", "highprice", "h", "high_price", "max", "haut"],
    "low":    ["low", "lowprice", "l", "low_price", "min", "bas"],
    "close":  ["close", "closeprice", "c", "close_price", "adjclose",
               "adj_close", "cloture", "clôture", "price", "last"],
    "volume": ["volume", "vol", "v", "tickvolume", "tick_volume",
               "realvolume", "quantity"],
}

DATETIME_ALIASES = ["datetime", "date_time", "timestamp", "time", "date",
                    "gmttime", "gmt_time", "localtime", "local_time", "dt"]

DATE_ONLY_ALIASES = ["date", "day", "jour"]
TIME_ONLY_ALIASES = ["time", "heure", "hour"]


def _normalise(name: str) -> str:
    """Ramène un nom de colonne à sa forme comparable."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch == "_")


def _find_column(columns: dict[str, str], candidates: List[str]) -> Optional[str]:
    """Cherche la première colonne dont le nom normalisé figure dans la liste."""
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


STANDARD_HEADER = ["time", "open", "high", "low", "close", "volume"]


def _looks_like_data(names: List[str]) -> bool:
    """Les « noms de colonnes » sont-ils en réalité une ligne de données ?

    Beaucoup d'exports (MetaTrader notamment) n'écrivent aucun en-tête. Sans
    détection, pandas prendrait la première bougie pour les noms de colonnes :
    on perdrait une ligne ET toute la détection échouerait.

    Deux indices : au moins la moitié des noms sont numériques, ou le premier
    s'analyse comme une date.
    """
    if not names:
        return False

    numeric = 0
    for name in names:
        try:
            float(str(name).replace(",", "."))
            numeric += 1
        except ValueError:
            pass
    if numeric >= max(1, len(names) // 2):
        return True

    first = pd.to_datetime(str(names[0]), errors="coerce", format="mixed")
    return not pd.isna(first)


def _read_any_csv(path: Path) -> pd.DataFrame:
    """Lit un CSV en devinant le séparateur ET la présence d'un en-tête.

    `sep=None` + `engine="python"` active le renifleur de séparateur de
    pandas : virgule, point-virgule ou tabulation sans configuration.
    Beaucoup d'exports MetaTrader utilisent le point-virgule, ce qui produit
    sinon un DataFrame d'une seule colonne — erreur très déroutante.
    """
    frame = pd.read_csv(path, sep=None, engine="python")

    if _looks_like_data(list(frame.columns)):
        # Relecture sans en-tête, en nommant les colonnes selon la convention
        # OHLCV standard : Date, Open, High, Low, Close, Volume.
        frame = pd.read_csv(path, sep=None, engine="python", header=None)
        count = frame.shape[1]
        if count < 5:
            raise ValueError(
                f"{path.name} : fichier sans en-tête et seulement {count} "
                "colonnes. Il en faut au moins 5 (date, O, H, L, C).")
        names = STANDARD_HEADER[:count]
        # Colonnes supplémentaires éventuelles (spread, tick count…) ignorées.
        names += [f"extra_{i}" for i in range(count - len(names))]
        frame.columns = names
        print("  aucun en-tête détecté -> colonnes supposées "
              f"{', '.join(STANDARD_HEADER[:min(count, 6)])}")

    return frame


def _disorder_score(times: pd.Series) -> float:
    """Proportion de reculs dans le temps, plus les dates illisibles.

    Un fichier correctement analysé est chronologique : le score vaut ~0.
    Une inversion jour/mois brise cet ordre sur environ 12 jours par mois,
    ce qui fait bondir le score.
    """
    if times.isna().all():
        return 2.0
    valid = times.dropna()
    if len(valid) < 2:
        return 1.0
    backwards = (valid.diff().dropna() < pd.Timedelta(0)).mean()
    missing = times.isna().mean()
    return float(backwards) + float(missing)


def _parse_times(stamps: pd.Series) -> pd.Series:
    """Analyse les horodatages en devinant la convention jour/mois."""
    sample = stamps.head(3000)

    candidates = []
    for dayfirst in (False, True):
        parsed = pd.to_datetime(sample, format="mixed", dayfirst=dayfirst,
                                errors="coerce")
        candidates.append((_disorder_score(parsed), dayfirst))

    candidates.sort()
    best_score, best_dayfirst = candidates[0]
    worst_score = candidates[1][0]

    # On ne signale le choix que s'il est réellement discriminant : sur un
    # format ISO (2024-01-03), les deux conventions donnent le même résultat.
    if worst_score - best_score > 0.05:
        convention = "jour/mois" if best_dayfirst else "mois/jour"
        print(f"  convention de date détectée : {convention}")

    return pd.to_datetime(stamps, format="mixed", dayfirst=best_dayfirst,
                          errors="coerce")


def load_ohlcv(source: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Charge un fichier CSV, ou tous les CSV d'un dossier.

    Renvoie un DataFrame normalisé aux colonnes :
        time, open, high, low, close, volume
    trié chronologiquement et dédoublonné.
    """
    source = Path(source)

    if source.is_dir():
        files = sorted(source.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"Aucun fichier correspondant à {pattern} dans {source.resolve()}")
        print(f"{len(files)} fichier(s) : {files[0].name} → {files[-1].name}")
        frames = [_read_any_csv(f) for f in files]
        raw = pd.concat(frames, ignore_index=True)
    else:
        if not source.exists():
            raise FileNotFoundError(f"Fichier introuvable : {source.resolve()}")
        print(f"Fichier : {source.name}")
        raw = _read_any_csv(source)

    return normalise_frame(raw)


def normalise_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Détecte les colonnes et produit un DataFrame au format interne."""
    columns = {_normalise(c): c for c in raw.columns}

    # ---------------- horodatage ----------------
    time_column = _find_column(columns, DATETIME_ALIASES)
    date_column = _find_column(columns, DATE_ONLY_ALIASES)
    hour_column = _find_column(columns, TIME_ONLY_ALIASES)

    if date_column and hour_column and date_column != hour_column:
        # Format MetaTrader : Date et Time dans deux colonnes séparées.
        stamps = (raw[date_column].astype(str).str.strip() + " "
                  + raw[hour_column].astype(str).str.strip())
    elif time_column:
        stamps = raw[time_column]
    else:
        raise ValueError(
            "Aucune colonne d'horodatage reconnue. Colonnes présentes : "
            f"{list(raw.columns)}")

    # ---------------- analyse de la date ----------------
    # Piège majeur : 03/01/2024 vaut « 3 janvier » en Europe et « 1er mars »
    # aux États-Unis. Se fier à `dayfirst` seul est fragile — pandas l'ignore
    # dans certains cas avec format="mixed", et l'inversion ne provoque AUCUNE
    # erreur : elle se contente de mélanger vos bougies.
    #
    # Détection automatique : les fichiers de marché sont pratiquement
    # toujours écrits dans l'ordre chronologique. On analyse donc des deux
    # façons et on garde celle qui produit la série la mieux ordonnée.
    times = _parse_times(stamps)

    # ---------------- prix ----------------
    data = {"time": times}
    for field in ("open", "high", "low", "close"):
        column = _find_column(columns, ALIASES[field])
        if column is None:
            raise ValueError(
                f"Colonne '{field}' introuvable. Colonnes présentes : "
                f"{list(raw.columns)}")
        # Nettoyage : certains exports écrivent « 1 234,56 » ou « $1,234.56 ».
        series = raw[column]
        if series.dtype == object:
            series = (series.astype(str)
                      .str.replace(r"[^\d\.\-]", "", regex=True))
        data[field] = pd.to_numeric(series, errors="coerce")

    volume_column = _find_column(columns, ALIASES["volume"])
    if volume_column is not None:
        data["volume"] = pd.to_numeric(raw[volume_column], errors="coerce")
    else:
        # Or au comptant, forex : pas de volume. On met zéro plutôt que de
        # planter — la feature correspondante sera désactivée en amont.
        data["volume"] = 0.0

    frame = pd.DataFrame(data)

    before = len(frame)
    frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
    frame = frame.drop_duplicates(subset="time").sort_values("time")
    frame = frame.reset_index(drop=True)

    dropped = before - len(frame)
    if dropped:
        print(f"  {dropped} ligne(s) écartée(s) : date illisible ou doublon")

    return frame


def frame_to_quotes(frame: pd.DataFrame, verbose: bool = True) -> List[Quote]:
    """Convertit le DataFrame normalisé en objets Quote, avec contrôle qualité."""
    quotes: List[Quote] = []
    rejected = {"incoherent": 0, "prix_nul": 0}

    for row in frame.itertuples(index=False):
        low, high, close, open_ = row.low, row.high, row.close, row.open

        if close <= 0 or high <= 0 or low <= 0:
            rejected["prix_nul"] += 1
            continue
        # Contrôle d'intégrité OHLC : le plus haut doit dominer, le plus bas
        # doit être en dessous. Une ligne qui viole cela est corrompue et
        # fausserait le stochastique, qui repose justement sur high/low.
        if not (low <= min(open_, close) and high >= max(open_, close)):
            rejected["incoherent"] += 1
            continue

        quotes.append(Quote(
            time=row.time.to_pydatetime(),
            open_price=float(open_),
            high_price=float(high),
            low_price=float(low),
            close_price=float(close),
            volume=float(row.volume) if not np.isnan(row.volume) else 0.0,
        ))

    if verbose:
        total_rejected = sum(rejected.values())
        if total_rejected:
            print(f"  {total_rejected} bougie(s) écartée(s) "
                  f"(OHLC incohérent : {rejected['incoherent']}, "
                  f"prix nul : {rejected['prix_nul']})")
    return quotes


def has_usable_volume(quotes: List[Quote]) -> bool:
    """Le volume porte-t-il une information exploitable ?

    Sur l'or au comptant et le forex, la colonne est absente ou constante.
    Une feature constante n'apporte rien au modèle et gaspille une entrée ;
    pire, elle donne l'illusion d'une information qui n'existe pas.
    """
    volumes = np.array([q.volume for q in quotes[:5000]], dtype=np.float64)
    if volumes.size == 0:
        return False
    return bool(volumes.std() > 0 and volumes.mean() > 0)