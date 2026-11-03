import numpy as np
import pytest

from app import analysis
from app.analysis import AnalysisError


def pulse(t, center, width=0.05):
    return np.exp(-(((t - center) / width) ** 2))


# --- outils de base ----------------------------------------------------------
def test_sample_interval_uses_median_and_ignores_gaps():
    t = np.concatenate([np.arange(0, 1, 0.01), np.arange(5, 6, 0.01)])
    assert analysis.sample_interval(t) == pytest.approx(0.01)


def test_sample_interval_needs_two_points():
    with pytest.raises(AnalysisError):
        analysis.sample_interval([1.0])


def test_signal_stats_frequency_counts_intervals():
    t = np.arange(0, 1.0001, 0.01)  # 101 points sur 1 s => 100 Hz
    stats = analysis.signal_stats(t)
    assert stats.num_points == 101
    assert stats.frequency == pytest.approx(100.0)


def test_signal_stats_single_point_has_no_frequency():
    assert analysis.signal_stats([3.0]).frequency is None


def test_array_channels_are_rejected():
    with pytest.raises(AnalysisError):
        analysis.auto_correlation(np.arange(4.0), np.zeros((4, 2)))


# --- rééchantillonnage --------------------------------------------------------
def test_resample_common_uses_overlap_and_finest_step():
    t1 = np.arange(0, 10, 0.01)
    t2 = np.arange(2, 12, 0.1)
    grid, r1, r2, step = analysis.resample_common(t1, t1 * 2, t2, t2 * 3)
    assert step == pytest.approx(0.01)
    assert grid[0] == pytest.approx(2.0)
    assert grid[-1] <= t1[-1] + 1e-9
    np.testing.assert_allclose(r1, grid * 2, rtol=1e-9)
    np.testing.assert_allclose(r2, grid * 3, rtol=1e-9)


def test_resample_common_without_overlap_raises():
    with pytest.raises(AnalysisError, match="recouvrent"):
        analysis.resample_common([0, 1], [0, 1], [2, 3], [0, 1])


def test_categorical_resampling_holds_last_value():
    grid, r1, _, _ = analysis.resample_common(
        [0.0, 1.0, 2.0], np.array(["A", "B", "C"]), np.arange(0, 2.01, 0.5), np.zeros(5)
    )
    assert list(r1) == ["A", "A", "B", "B", "C"]


def test_nan_samples_are_ignored():
    t = np.arange(0, 1, 0.01)
    s = np.sin(2 * np.pi * 3 * t)
    s[10] = np.nan
    lags, ac = analysis.auto_correlation(t, s)
    assert np.all(np.isfinite(ac))


# --- cross-corrélation ----------------------------------------------------------
def test_cross_correlation_finds_known_delay():
    t = np.arange(0, 10, 0.01)
    s1 = pulse(t, 5.5)  # signal 1 en retard de 0,5 s
    s2 = pulse(t, 5.0)
    lags, corr = analysis.cross_correlation(t, s1, t, s2)
    assert lags[np.argmax(corr)] == pytest.approx(0.5, abs=0.011)
    assert corr.max() <= 1.0 + 1e-9


def test_cross_correlation_with_different_sampling_rates():
    # Même signal, 100 Hz contre 10 Hz, avec un décalage de 1 s
    t1 = np.arange(0, 20, 0.01)
    t2 = np.arange(0, 20, 0.1)
    lags, corr = analysis.cross_correlation(t1, pulse(t1, 9.0, 0.3), t2, pulse(t2, 10.0, 0.3))
    assert lags[np.argmax(corr)] == pytest.approx(-1.0, abs=0.02)


def test_cross_correlation_at_zero_lag_equals_pearson():
    rng = np.random.default_rng(0)
    t = np.arange(0, 5, 0.01)
    a = rng.normal(size=t.size)
    b = a + rng.normal(size=t.size)
    lags, corr = analysis.cross_correlation(t, a, t, b)
    expected = np.corrcoef(a, b)[0, 1]
    assert corr[np.argmin(np.abs(lags))] == pytest.approx(expected, rel=1e-6)


def test_cross_correlation_of_constant_signal_raises():
    t = np.arange(0, 1, 0.01)
    with pytest.raises(AnalysisError, match="constant"):
        analysis.cross_correlation(t, np.ones_like(t), t, t)


def test_cross_correlation_categorical_finds_delay():
    t = np.arange(0, 10, 0.01)
    s2 = np.where((t > 3) & (t < 4), "ON", "OFF")
    s1 = np.where((t > 3.2) & (t < 4.2), "ON", "OFF")
    lags, corr = analysis.cross_correlation(t, s1, t, s2)
    assert lags[np.argmax(corr)] == pytest.approx(0.2, abs=0.011)


def test_cross_correlation_mixed_types_raises():
    t = np.arange(0, 1, 0.01)
    with pytest.raises(AnalysisError, match="numérique"):
        analysis.cross_correlation(t, t, t, np.array(["A"] * t.size))


# --- auto-corrélation ---------------------------------------------------------
def test_auto_correlation_of_periodic_signal():
    t = np.arange(0, 10, 0.01)
    lags, ac = analysis.auto_correlation(t, np.sin(2 * np.pi * 2 * t))  # période 0,5 s
    assert lags[0] == 0 and ac[0] == pytest.approx(1.0)
    window = (lags > 0.3) & (lags < 0.7)
    assert lags[window][np.argmax(ac[window])] == pytest.approx(0.5, abs=0.011)


def test_auto_correlation_categorical():
    t = np.arange(0, 10, 0.01)
    states = np.where(np.sin(2 * np.pi * t) > 0, "HAUT", "BAS")  # période 1 s
    lags, ac = analysis.auto_correlation(t, states)
    window = (lags > 0.7) & (lags < 1.3)
    assert lags[window][np.argmax(ac[window])] == pytest.approx(1.0, abs=0.02)


def test_auto_correlation_constant_raises():
    with pytest.raises(AnalysisError):
        analysis.auto_correlation(np.arange(10.0), np.ones(10))


# --- Pearson / concordance ----------------------------------------------------
def test_pearson_aligns_signals_in_time():
    # Mêmes données, fréquences différentes : Pearson doit valoir ~1
    t1 = np.arange(0, 10, 0.01)
    t2 = np.arange(0, 10, 0.05)
    res = analysis.pearson_correlation(t1, np.sin(2 * np.pi * 0.3 * t1), t2, np.sin(2 * np.pi * 0.3 * t2))
    assert res.kind == "pearson"
    assert res.value == pytest.approx(1.0, abs=1e-3)


def test_pearson_anticorrelated():
    t = np.arange(0, 10, 0.01)
    assert analysis.pearson_correlation(t, t, t, -t).value == pytest.approx(-1.0)


def test_categorical_similarity_is_agreement_rate():
    t = np.arange(0, 4.0, 1.0)
    res = analysis.pearson_correlation(t, np.array(["A", "B", "A", "B"]), t, np.array(["A", "B", "B", "B"]))
    assert res.kind == "concordance"
    assert res.value == pytest.approx(0.75)
    assert "75" in res.describe()


def test_boolean_signals_are_numeric():
    t = np.arange(0, 1, 0.01)
    b = np.sin(2 * np.pi * 3 * t) > 0
    assert analysis.pearson_correlation(t, b, t, b).value == pytest.approx(1.0)
