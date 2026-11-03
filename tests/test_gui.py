"""Test de fumée de l'interface : nécessite un affichage (ex. xvfb-run)."""

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Pas d'affichage disponible")
    from app.gui import App

    application = App(root)
    yield application
    application.close()


def wait_idle(app, timeout=10.0):
    import time

    end = time.time() + timeout
    while app.busy and time.time() < end:
        app.root.update()
        time.sleep(0.01)
    app.root.update()
    assert not app.busy


def test_buttons_without_files_warn_instead_of_crashing(app, monkeypatch):
    warnings = []
    monkeypatch.setattr("app.gui.messagebox.showwarning", lambda *a, **k: warnings.append(a))
    app.run_all()
    app.compute_pearson()
    app.mdf4.plot_channel()
    assert len(warnings) == 3
    assert "MDF4" in warnings[0][1]


def test_full_analysis_with_fake_sources(app, monkeypatch):
    from app.mdf_io import ChannelData

    class FakeSource:
        def __init__(self, step, delay):
            t = np.arange(0, 10, step)
            self.channels = ["Vitesse"]
            self.data = ChannelData("Vitesse", t, np.exp(-(((t - 5 - delay) / 0.3) ** 2)), "km/h")

        def get(self, name):
            return self.data

        def close(self):
            pass

    for panel, src in ((app.mdf4, FakeSource(0.02, 0.2)), (app.mdf3, FakeSource(0.01, 0.0))):
        panel.source = src
        panel.channel_var.set("Vitesse")

    errors = []
    monkeypatch.setattr("app.gui.messagebox.showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr("app.gui.messagebox.showwarning", lambda *a, **k: errors.append(a))
    app.run_all()
    wait_idle(app)
    assert errors == []
    assert "Pearson" in app.result_var.get()
    assert app.xcorr_plot.ax.get_title().startswith("Cross-Corr")

    app.reset()
    assert app.result_var.get() == ""


def test_computation_errors_are_shown(app, monkeypatch):
    errors = []
    monkeypatch.setattr("app.gui.messagebox.showerror", lambda *a, **k: errors.append(a))

    def boom():
        raise RuntimeError("panne")

    app.run_task("Test", boom, lambda r: None)
    wait_idle(app)
    assert errors and "panne" in errors[0][1]
    assert "Échec" in app.status_var.get()
