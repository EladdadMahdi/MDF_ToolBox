"""Auto-test de l'application packagée (``MDF_ToolBox.exe --self-test``).

Utilisé par la CI pour vérifier que l'exécutable contient bien toutes ses
dépendances : écriture puis lecture de vrais fichiers MDF3/MDF4, calculs
d'analyse et création de l'interface. Code de sortie 0 si tout va bien.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path


def _check_mdf_and_analysis(directory: Path) -> None:
    import numpy as np
    from asammdf import MDF, Signal

    from . import analysis
    from .mdf_io import MdfSource

    t = np.arange(0, 10, 0.01)
    for version, ext in (("3.30", "mdf"), ("4.10", "mf4")):
        mdf = MDF(version=version)
        mdf.append([Signal(np.sin(2 * np.pi * 0.5 * t), t, name="Signal", unit="V")])
        mdf.save(directory / f"selftest.{ext}", overwrite=True)
        mdf.close()

    s3, s4 = MdfSource(directory / "selftest.mdf"), MdfSource(directory / "selftest.mf4")
    try:
        a, b = s4.get("Signal"), s3.get("Signal")
        result = analysis.pearson_correlation(a.timestamps, a.samples, b.timestamps, b.samples)
        if abs(result.value - 1.0) > 1e-6:
            raise AssertionError(f"Pearson inattendu : {result.value}")
        analysis.cross_correlation(a.timestamps, a.samples, b.timestamps, b.samples)
        analysis.auto_correlation(a.timestamps, a.samples)
    finally:
        s3.close()
        s4.close()


def _check_gui() -> str:
    import tkinter as tk

    from .gui import App, create_root

    try:
        root = create_root()
    except tk.TclError as exc:  # pas d'écran : on ne peut pas tester l'interface
        return f"Interface non testée (pas d'affichage) : {exc}"
    app = App(root)
    root.update()
    app.close()
    return "Interface OK"


#: Rapport écrit à chaque exécution : l'exécutable Windows n'a pas de console.
REPORT_FILE = Path(tempfile.gettempdir()) / "MDF_ToolBox_selftest.log"


def run() -> int:
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        code = _run(report)
    if sys.stdout is not None:
        print(REPORT_FILE.read_text(encoding="utf-8"), end="")
    return code


def _run(report) -> int:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _check_mdf_and_analysis(Path(tmp))
        report.write("Lecture MDF3/MDF4 et calculs OK\n")
        report.write(_check_gui() + "\n")
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=report)
        report.write("Auto-test ÉCHOUÉ\n")
        return 1
    report.write("Auto-test OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
