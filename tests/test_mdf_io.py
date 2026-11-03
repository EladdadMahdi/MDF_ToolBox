import numpy as np
import pytest

asammdf = pytest.importorskip("asammdf")

from app import analysis  # noqa: E402
from app.mdf_io import MdfSource  # noqa: E402


@pytest.fixture(scope="module")
def mdf_files(tmp_path_factory):
    """Crée une même mesure en MDF3 (100 Hz) et en MDF4 (50 Hz, décalée de 0,2 s)."""
    directory = tmp_path_factory.mktemp("mdf")
    paths = {}
    for version, step, delay, ext in (("3.30", 0.01, 0.0, "mdf"), ("4.10", 0.02, 0.2, "mf4")):
        t = np.arange(0, 10, step)
        states = np.where(np.sin(2 * np.pi * 0.5 * (t - delay)) > 0, b"ON", b"OFF")
        mdf = asammdf.MDF(version=version)
        mdf.append(
            [
                asammdf.Signal(np.exp(-(((t - 5 - delay) / 0.3) ** 2)), t, name="Vitesse", unit="km/h"),
                asammdf.Signal(states, t, name="Etat", encoding="latin-1"),
            ]
        )
        path = directory / f"mesure.{ext}"
        mdf.save(path, overwrite=True)
        mdf.close()
        paths[version[0]] = path
    return paths


def test_channels_are_listed_without_time(mdf_files):
    src = MdfSource(mdf_files["3"])
    try:
        assert src.channels == ["Etat", "Vitesse"]
        assert src.version.startswith("3")
    finally:
        src.close()


def test_get_reads_unit_and_caches(mdf_files):
    src = MdfSource(mdf_files["4"])
    try:
        ch = src.get("Vitesse")
        assert ch.unit == "km/h"
        assert src.get("Vitesse") is ch
        with pytest.raises(KeyError):
            src.get("Inconnue")
    finally:
        src.close()


def test_text_channels_are_comparable_between_versions(mdf_files):
    s3, s4 = MdfSource(mdf_files["3"]), MdfSource(mdf_files["4"])
    try:
        e3, e4 = s3.get("Etat"), s4.get("Etat")
        assert set(np.unique(e3.samples)) == {"ON", "OFF"} == set(np.unique(e4.samples))

        v3, v4 = s3.get("Vitesse"), s4.get("Vitesse")
        lags, corr = analysis.cross_correlation(v4.timestamps, v4.samples, v3.timestamps, v3.samples)
        assert lags[np.argmax(corr)] == pytest.approx(0.2, abs=0.011)

        lags, corr = analysis.cross_correlation(e4.timestamps, e4.samples, e3.timestamps, e3.samples)
        assert lags[np.argmax(corr)] == pytest.approx(0.2, abs=0.03)
    finally:
        s3.close()
        s4.close()
