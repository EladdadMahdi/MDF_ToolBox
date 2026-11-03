"""Lecture des fichiers MDF (versions 3 et 4) via asammdf."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from asammdf import MDF, set_global_option

# Une voie peut apparaître dans plusieurs groupes : on prend la première
# occurrence au lieu de lever une exception.
set_global_option("raise_on_multiple_occurrences", False)

#: Filtres proposés dans la boîte de dialogue d'ouverture. Les extensions ne
#: préjugent pas de la version : asammdf détecte la version dans le fichier.
MDF_FILETYPES = [
    ("Fichiers MDF", "*.mf4 *.mdf *.dat *.MF4 *.MDF *.DAT"),
    ("Tous les fichiers", "*.*"),
]


@dataclass(frozen=True)
class ChannelData:
    name: str
    timestamps: np.ndarray
    samples: np.ndarray
    unit: str


def _normalize_samples(samples: np.ndarray) -> np.ndarray:
    """Convertit les textes (octets) en chaînes Python propres.

    MDF3 et MDF4 ne stockent pas les textes de la même façon (octets bruts,
    bourrage par des caractères nuls...). Sans cette normalisation, un même
    état « ON » ne serait pas reconnu identique entre les deux fichiers.
    """
    if samples.dtype.kind == "S" or samples.dtype == object:

        def to_str(v):
            if isinstance(v, bytes):
                v = v.decode("latin-1", errors="replace")
            return str(v).strip("\x00 ").strip()

        return np.array([to_str(v) for v in samples], dtype=object).astype(str)
    if samples.dtype.kind == "U":
        return np.char.strip(samples)
    return samples


class MdfSource:
    """Fichier MDF ouvert, avec cache des voies déjà lues."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mdf = MDF(str(self.path))
        self.version = str(self._mdf.version)
        self.channels = sorted(name for name in self._mdf.channels_db if name and name.lower() != "time")
        self._cache: dict[str, ChannelData] = {}

    def get(self, name: str) -> ChannelData:
        if name not in self._cache:
            if name not in self._mdf.channels_db:
                raise KeyError(f"La voie « {name} » n'existe pas dans {self.path.name}.")
            signal = self._mdf.get(name)
            self._cache[name] = ChannelData(
                name=name,
                timestamps=np.asarray(signal.timestamps, dtype=np.float64),
                samples=_normalize_samples(np.asarray(signal.samples)),
                unit=str(getattr(signal, "unit", "") or ""),
            )
        return self._cache[name]

    def close(self) -> None:
        self._cache.clear()
        self._mdf.close()
