"""Fonctions d'analyse de signaux, indépendantes de l'interface graphique.

Toutes les fonctions travaillent sur des tableaux NumPy ``(timestamps, samples)``
et sont donc testables sans fichier MDF ni affichage.

Conventions
-----------
* Les signaux de deux fichiers différents n'ont en général ni la même
  fréquence d'échantillonnage ni le même instant de départ : avant toute
  comparaison, ils sont rééchantillonnés sur une base de temps commune
  (:func:`resample_common`), limitée à la plage temporelle où les deux
  signaux existent.
* Signaux numériques : interpolation linéaire.
  Signaux catégoriels (textes, états) : maintien de la dernière valeur
  (« zero-order hold »), car interpoler entre deux états n'a pas de sens.
* Pour la corrélation croisée, un retard positif signifie que le signal 1
  est en retard sur le signal 2 (il faut le décaler vers la gauche pour
  qu'il coïncide avec le signal 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

#: Nombre maximal de points de la base de temps commune. Au-delà, le pas est
#: élargi pour borner la mémoire et le temps de calcul.
MAX_POINTS = 2_000_000


class AnalysisError(ValueError):
    """Erreur de calcul présentable telle quelle à l'utilisateur."""


# ---------------------------------------------------------------------------
# Outils de base
# ---------------------------------------------------------------------------
def is_numeric(samples: np.ndarray) -> bool:
    """Vrai si les échantillons sont numériques (booléens inclus)."""
    samples = np.asarray(samples)
    return np.issubdtype(samples.dtype, np.number) or samples.dtype == np.bool_


def _as_1d(timestamps, samples) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(timestamps, dtype=np.float64)
    s = np.asarray(samples)
    if s.ndim != 1:
        raise AnalysisError(
            "Les voies de type tableau (plusieurs valeurs par instant) ne sont pas prises en charge."
        )
    if t.shape != s.shape:
        raise AnalysisError("Les horodatages et les échantillons n'ont pas la même longueur.")
    if s.dtype == np.bool_:
        s = s.astype(np.float64)
    return t, s


def _clean(timestamps, samples) -> tuple[np.ndarray, np.ndarray]:
    """Convertit, retire les points NaN et trie par temps croissant."""
    t, s = _as_1d(timestamps, samples)
    mask = np.isfinite(t)
    if is_numeric(s):
        mask &= np.isfinite(s)
    t, s = t[mask], s[mask]
    if t.size and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, s = t[order], s[order]
    return t, s


def sample_interval(timestamps) -> float:
    """Période d'échantillonnage typique (médiane des écarts, robuste aux trous)."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size < 2:
        raise AnalysisError("Il faut au moins 2 points pour calculer une période d'échantillonnage.")
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise AnalysisError("Tous les horodatages sont identiques.")
    return float(np.median(diffs))


@dataclass(frozen=True)
class SignalStats:
    num_points: int
    duration: float
    frequency: float | None  # Hz, None si non calculable


def signal_stats(timestamps) -> SignalStats:
    """Nombre de points, durée et fréquence moyenne d'échantillonnage."""
    t = np.asarray(timestamps, dtype=np.float64)
    n = int(t.size)
    if n < 2:
        return SignalStats(n, 0.0, None)
    duration = float(t[-1] - t[0])
    # n points couvrent (n - 1) intervalles
    frequency = (n - 1) / duration if duration > 0 else None
    return SignalStats(n, duration, frequency)


def encode_categories(samples, categories=None) -> tuple[np.ndarray, np.ndarray]:
    """Encodage « one-hot » : une colonne indicatrice (0/1) par catégorie."""
    samples = np.asarray(samples)
    if categories is None:
        categories = np.unique(samples)
    indicators = (samples[:, None] == np.asarray(categories)[None, :]).astype(np.float64)
    return indicators, np.asarray(categories)


def _zscore(x: np.ndarray) -> np.ndarray | None:
    """Centre-réduit ``x`` ; renvoie None si le signal est constant."""
    std = np.std(x)
    if not np.isfinite(std) or std == 0:
        return None
    return (x - np.mean(x)) / std


# ---------------------------------------------------------------------------
# Rééchantillonnage
# ---------------------------------------------------------------------------
def _grid(start: float, stop: float, step: float) -> np.ndarray:
    # La petite tolérance évite de perdre le dernier point à cause des arrondis
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    if n > MAX_POINTS:
        n = MAX_POINTS
    return start + step * np.arange(n)


def _resample(t: np.ndarray, s: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if is_numeric(s):
        return np.interp(grid, t, s.astype(np.float64))
    # Maintien de la dernière valeur connue
    idx = np.searchsorted(t, grid, side="right") - 1
    return s[np.clip(idx, 0, t.size - 1)]


def resample_uniform(timestamps, samples) -> tuple[np.ndarray, np.ndarray, float]:
    """Rééchantillonne un signal sur une grille régulière à sa période typique."""
    t, s = _clean(timestamps, samples)
    if t.size < 2:
        raise AnalysisError("Le signal contient moins de 2 points exploitables.")
    step = sample_interval(t)
    if (t[-1] - t[0]) / step + 1 > MAX_POINTS:
        step = (t[-1] - t[0]) / (MAX_POINTS - 1)
    grid = _grid(t[0], t[-1], step)
    return grid, _resample(t, s, grid), step


def resample_common(t1, s1, t2, s2):
    """Rééchantillonne deux signaux sur une base de temps commune.

    La grille couvre l'intervalle où les deux signaux sont définis, avec la
    plus fine des deux périodes d'échantillonnage.

    Renvoie ``(grid, s1_resampled, s2_resampled, step)``.
    """
    t1, s1 = _clean(t1, s1)
    t2, s2 = _clean(t2, s2)
    if t1.size < 2 or t2.size < 2:
        raise AnalysisError("Chaque signal doit contenir au moins 2 points exploitables.")
    start = max(t1[0], t2[0])
    stop = min(t1[-1], t2[-1])
    if stop <= start:
        raise AnalysisError(
            "Les deux signaux ne se recouvrent pas dans le temps "
            f"([{t1[0]:.3f} ; {t1[-1]:.3f}] s et [{t2[0]:.3f} ; {t2[-1]:.3f}] s)."
        )
    step = min(sample_interval(t1), sample_interval(t2))
    if (stop - start) / step + 1 > MAX_POINTS:
        step = (stop - start) / (MAX_POINTS - 1)
    grid = _grid(start, stop, step)
    if grid.size < 2:
        raise AnalysisError("La plage temporelle commune est trop courte.")
    return grid, _resample(t1, s1, grid), _resample(t2, s2, grid), step


# ---------------------------------------------------------------------------
# Corrélations
# ---------------------------------------------------------------------------
def _xcorr_normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Corrélation croisée de deux signaux centrés-réduits, divisée par N.

    Au retard nul, la valeur obtenue est exactement le coefficient de Pearson.
    """
    return scipy.signal.correlate(a, b, mode="full", method="auto") / a.size


def cross_correlation(t1, s1, t2, s2) -> tuple[np.ndarray, np.ndarray]:
    """Corrélation croisée normalisée entre deux signaux.

    * numérique / numérique : signaux centrés-réduits, résultat dans [-1, 1].
    * catégoriel / catégoriel : moyenne des corrélations des indicatrices des
      états communs aux deux signaux.
    * mélange numérique / catégoriel : non défini, une erreur est levée.

    Renvoie ``(lags_en_secondes, correlation)``.
    """
    _, r1, r2, step = resample_common(t1, s1, t2, s2)
    num1, num2 = is_numeric(r1), is_numeric(r2)

    if num1 and num2:
        a, b = _zscore(r1), _zscore(r2)
        if a is None or b is None:
            raise AnalysisError("Un des signaux est constant sur la plage commune : corrélation non définie.")
        corr = _xcorr_normalized(a, b)
    elif not num1 and not num2:
        common = np.intersect1d(np.unique(r1), np.unique(r2))
        curves = []
        if common.size:
            ind1, _ = encode_categories(r1, common)
            ind2, _ = encode_categories(r2, common)
            for k in range(common.size):
                a, b = _zscore(ind1[:, k]), _zscore(ind2[:, k])
                if a is not None and b is not None:
                    curves.append(_xcorr_normalized(a, b))
        if not curves:
            raise AnalysisError(
                "Les deux signaux n'ont aucun état variable en commun : corrélation non définie."
            )
        corr = np.mean(curves, axis=0)
    else:
        raise AnalysisError("Corrélation impossible entre un signal numérique et un signal textuel (états).")

    lags = scipy.signal.correlation_lags(r1.size, r2.size, mode="full") * step
    return lags, corr


def auto_correlation(timestamps, samples) -> tuple[np.ndarray, np.ndarray]:
    """Auto-corrélation normalisée (valeur 1 au retard nul), retards positifs.

    Le signal est d'abord rééchantillonné sur une grille régulière pour que
    l'axe des retards soit exact même si l'acquisition présente des trous.
    """
    _, s, step = resample_uniform(timestamps, samples)

    if is_numeric(s):
        z = _zscore(s)
        if z is None:
            raise AnalysisError("Le signal est constant : auto-corrélation non définie.")
        columns = [z]
    else:
        indicators, _ = encode_categories(s)
        columns = [c for c in (_zscore(col) for col in indicators.T) if c is not None]
        if not columns:
            raise AnalysisError("Le signal reste dans un seul état : auto-corrélation non définie.")

    ac = np.mean([scipy.signal.correlate(c, c, mode="full", method="auto") for c in columns], axis=0)
    ac = ac[s.size - 1 :]
    ac = ac / ac[0]
    lags = np.arange(ac.size) * step
    return lags, ac


@dataclass(frozen=True)
class SimilarityResult:
    """Résultat de comparaison point à point de deux signaux."""

    kind: str  # "pearson" ou "concordance"
    value: float
    num_points: int

    def describe(self) -> str:
        if self.kind == "pearson":
            return f"Coefficient de Pearson : {self.value:.3f}  ({self.num_points} points)"
        return f"Taux de concordance des états : {self.value:.1%}  ({self.num_points} points)"


def pearson_correlation(t1, s1, t2, s2) -> SimilarityResult:
    """Compare deux signaux instant par instant sur leur plage commune.

    * numérique / numérique : coefficient de Pearson.
    * catégoriel / catégoriel : pourcentage du temps où les états sont égaux
      (le coefficient de Pearson n'a pas de sens sur des états non ordonnés).
    """
    _, r1, r2, _ = resample_common(t1, s1, t2, s2)
    num1, num2 = is_numeric(r1), is_numeric(r2)

    if num1 and num2:
        a, b = _zscore(r1), _zscore(r2)
        if a is None or b is None:
            raise AnalysisError(
                "Un des signaux est constant sur la plage commune : coefficient de Pearson non défini."
            )
        value = float(np.clip(np.mean(a * b), -1.0, 1.0))
        return SimilarityResult("pearson", value, int(r1.size))
    if not num1 and not num2:
        return SimilarityResult("concordance", float(np.mean(r1 == r2)), int(r1.size))
    raise AnalysisError("Comparaison impossible entre un signal numérique et un signal textuel (états).")
