"""Interface graphique Tkinter de comparaison de voies MDF3 / MDF4."""

from __future__ import annotations

import logging
import sys
import tempfile
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import __version__, analysis
from .mdf_io import MDF_FILETYPES, ChannelData, MdfSource

log = logging.getLogger(__name__)

APP_TITLE = f"MDF Channel Visualization v{__version__}"
POLL_MS = 50


def format_stats(channel: ChannelData) -> str:
    stats = analysis.signal_stats(channel.timestamps)
    freq = f"{stats.frequency:.1f} Hz" if stats.frequency is not None else "n/a"
    return f"Nombre de points : {stats.num_points}\nFréquence : {freq}"


class PlotArea(ttk.Frame):
    """Un graphique matplotlib avec sa barre d'outils (zoom, déplacement...)."""

    def __init__(self, master, title: str):
        super().__init__(master)
        self.empty_title = title
        self.figure = Figure(figsize=(4, 3), dpi=90, layout="constrained")
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.clear()

    def clear(self) -> None:
        self.ax.clear()
        self.ax.set_title(self.empty_title, fontsize=10)
        self.canvas.draw_idle()

    def show(self, title: str, xlabel: str, ylabel: str, draw: Callable, note: str | None = None):
        self.ax.clear()
        draw(self.ax)
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xlabel(xlabel, fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        self.ax.grid(True, alpha=0.3)
        if note:
            self.ax.annotate(
                note,
                xy=(0.01, 0.98),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=8,
                color="red",
                bbox=dict(facecolor="lightgrey", edgecolor="black", boxstyle="round,pad=0.2"),
            )
        self.canvas.draw_idle()


class FilePanel(ttk.LabelFrame):
    """Colonne « fichier » : import, choix de voie, tracé et auto-corrélation."""

    def __init__(self, master, app: App, label: str):
        super().__init__(master, text=f" Fichier {label} ", padding=4)
        self.app = app
        self.label = label
        self.source: MdfSource | None = None
        self.channel_var = tk.StringVar()
        self.file_var = tk.StringVar(value="Aucun fichier chargé")

        ttk.Button(self, text=f"Importer fichier {label}", command=self.load).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(self, textvariable=self.file_var, foreground="grey").grid(row=1, column=0, sticky="w")
        self.combo = ttk.Combobox(self, textvariable=self.channel_var, values=[])
        self.combo.grid(row=2, column=0, sticky="ew", pady=2)
        # Saisie au clavier = filtre sur la liste des voies
        self.combo.bind("<KeyRelease>", self._filter_channels)
        ttk.Button(self, text=f"Tracer voie {label}", command=self.plot_channel).grid(
            row=3, column=0, sticky="ew"
        )
        self.signal_plot = PlotArea(self, f"{label} - voie")
        self.signal_plot.grid(row=4, column=0, sticky="nsew", pady=2)
        ttk.Button(self, text=f"Tracer Auto-Corrélation {label}", command=self.plot_auto_correlation).grid(
            row=5, column=0, sticky="ew"
        )
        self.autocorr_plot = PlotArea(self, f"{label} - auto-corrélation")
        self.autocorr_plot.grid(row=6, column=0, sticky="nsew", pady=2)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(6, weight=1)

    # -- fichier / voie -----------------------------------------------------
    def load(self) -> None:
        path = filedialog.askopenfilename(
            title=f"Sélectionner le fichier {self.label}", filetypes=MDF_FILETYPES
        )
        if not path:
            return

        def done(source: MdfSource) -> None:
            if self.source is not None:
                self.source.close()
            self.source = source
            self.file_var.set(f"{source.path.name}  (MDF {source.version})")
            self.combo["values"] = source.channels
            self.channel_var.set(source.channels[0] if source.channels else "")
            self.clear()

        self.app.run_task(f"Chargement de {path}", lambda: MdfSource(path), done)

    def _filter_channels(self, event) -> None:
        if self.source is None or event.keysym in ("Up", "Down", "Return", "Escape"):
            return
        text = self.channel_var.get().lower()
        self.combo["values"] = [c for c in self.source.channels if text in c.lower()]

    def selected_channel(self) -> str:
        """Nom de la voie choisie ; lève une erreur lisible sinon."""
        if self.source is None:
            raise analysis.AnalysisError(f"Veuillez importer un fichier {self.label}.")
        name = self.channel_var.get().strip()
        if not name:
            raise analysis.AnalysisError(f"Veuillez choisir une voie dans le fichier {self.label}.")
        if name not in self.source.channels:
            raise analysis.AnalysisError(f"La voie « {name} » n'existe pas dans le fichier {self.label}.")
        return name

    def fetcher(self) -> Callable[[], ChannelData]:
        """À appeler dans le thread de l'interface : renvoie une fonction qui lit
        la voie choisie et peut s'exécuter dans le thread de calcul (le nom de
        la voie est capturé ici, car les variables Tk ne sont pas thread-safe)."""
        source, name = self.source, self.selected_channel()
        return lambda: source.get(name)

    # -- tracés ---------------------------------------------------------------
    def show_channel(self, ch: ChannelData) -> None:
        self.signal_plot.show(
            f"{self.label} - {ch.name}",
            "Temps (s)",
            ch.unit,
            lambda ax: ax.plot(ch.timestamps, ch.samples),
            note=format_stats(ch),
        )

    def show_auto_correlation(self, ch: ChannelData, lags, ac) -> None:
        self.autocorr_plot.show(
            f"{self.label} Auto-Corrélation - {ch.name}",
            "Retard (s)",
            "Auto-corrélation",
            lambda ax: ax.plot(lags, ac),
        )

    def plot_channel(self) -> None:
        if self.app.check_ready(self):
            self.app.run_task("Lecture de la voie", self.fetcher(), self.show_channel)

    def plot_auto_correlation(self) -> None:
        if not self.app.check_ready(self):
            return
        fetch = self.fetcher()

        def compute():
            ch = fetch()
            return ch, *analysis.auto_correlation(ch.timestamps, ch.samples)

        self.app.run_task("Calcul de l'auto-corrélation", compute, lambda r: self.show_auto_correlation(*r))

    def clear(self) -> None:
        self.signal_plot.clear()
        self.autocorr_plot.clear()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        root.title(APP_TITLE)
        root.protocol("WM_DELETE_WINDOW", self.close)

        # Fenêtre adaptée à l'écran et redimensionnable
        width = min(1400, root.winfo_screenwidth() - 40)
        height = min(980, root.winfo_screenheight() - 80)
        root.geometry(f"{width}x{height}")
        root.minsize(900, 600)

        main = ttk.Frame(root, padding=4)
        main.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self.mdf4 = FilePanel(main, self, "MDF4")
        self.mdf3 = FilePanel(main, self, "MDF3")
        self.mdf4.grid(row=0, column=0, sticky="nsew", padx=2)
        self.mdf3.grid(row=0, column=1, sticky="nsew", padx=2)
        self._build_comparison_panel(main).grid(row=0, column=2, sticky="nsew", padx=2)
        for col in range(3):
            main.columnconfigure(col, weight=1, uniform="cols")
        main.rowconfigure(0, weight=1)

        status = ttk.Frame(root, padding=(4, 0, 4, 4))
        status.grid(row=1, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=200)
        self.progress.pack(side="right")

    def _build_comparison_panel(self, master) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text=" Comparaison MDF4 / MDF3 ", padding=4)
        ttk.Button(frame, text="Calcul coefficient de Pearson", command=self.compute_pearson).grid(
            row=0, column=0, columnspan=3, sticky="ew"
        )
        self.result_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.result_var, foreground="red", font=("", 11, "bold")).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=2
        )
        ttk.Button(frame, text="Analyse complète", command=self.run_all).grid(row=2, column=0, sticky="ew")
        ttk.Button(frame, text="Cross-corrélation", command=self.plot_cross_correlation).grid(
            row=2, column=1, sticky="ew", padx=2
        )
        ttk.Button(frame, text="Réinitialiser", command=self.reset).grid(row=2, column=2, sticky="ew")
        self.xcorr_plot = PlotArea(frame, "Cross-corrélation MDF4 vs MDF3")
        self.xcorr_plot.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=2)
        ttk.Button(frame, text="Superposition MDF3 et MDF4", command=self.plot_superimposed).grid(
            row=4, column=0, columnspan=3, sticky="ew"
        )
        self.overlay_plot = PlotArea(frame, "Superposition MDF4 & MDF3")
        self.overlay_plot.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=2)
        for col in range(3):
            frame.columnconfigure(col, weight=1)
        frame.rowconfigure(3, weight=1)
        frame.rowconfigure(5, weight=1)
        return frame

    # -- exécution en arrière-plan -------------------------------------------
    def run_task(self, label: str, compute: Callable, on_done: Callable) -> None:
        """Exécute ``compute`` dans un thread, puis ``on_done(résultat)`` dans
        le thread de l'interface. Tkinter n'étant pas thread-safe, ``compute``
        ne doit jamais toucher aux widgets."""
        if self.busy:
            self.status_var.set("Un calcul est déjà en cours, veuillez patienter...")
            return
        self.busy = True
        self.status_var.set(label + "...")
        self.progress.start(10)
        future = self.executor.submit(compute)
        self.root.after(POLL_MS, self._poll, future, label, on_done)

    def _poll(self, future: Future, label: str, on_done: Callable) -> None:
        if not future.done():
            self.root.after(POLL_MS, self._poll, future, label, on_done)
            return
        self.busy = False
        self.progress.stop()
        try:
            result = future.result()
            on_done(result)
        except Exception as exc:  # noqa: BLE001 - toute erreur est montrée à l'utilisateur
            self.status_var.set(f"Échec : {label}")
            self.report_error(label, exc)
        else:
            self.status_var.set("Prêt")

    def report_error(self, context: str, exc: Exception) -> None:
        if isinstance(exc, analysis.AnalysisError):
            log.warning("%s : %s", context, exc)
        else:
            log.exception("%s", context, exc_info=exc)
        messagebox.showerror("Erreur", f"{context}\n\n{exc}", parent=self.root)

    def check_ready(self, *panels: FilePanel) -> bool:
        """Vérifie (dans le thread de l'interface) que fichiers et voies sont choisis."""
        try:
            for panel in panels:
                panel.selected_channel()
        except analysis.AnalysisError as exc:
            messagebox.showwarning("Avertissement", str(exc), parent=self.root)
            return False
        return True

    # -- comparaisons ---------------------------------------------------------
    def _fetch_both(self) -> Callable[[], tuple[ChannelData, ChannelData]]:
        f4, f3 = self.mdf4.fetcher(), self.mdf3.fetcher()
        return lambda: (f4(), f3())

    def show_cross_correlation(self, lags, corr) -> None:
        def draw(ax):
            ax.plot(lags, corr)
            ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
            peak = int(abs(corr).argmax())
            ax.plot(lags[peak], corr[peak], "o", color="orange")

        peak = int(abs(corr).argmax())
        self.xcorr_plot.show(
            "Cross-Corrélation MDF4 vs MDF3",
            "Retard (s)",
            "Cross-corrélation",
            draw,
            note=f"Pic : {corr[peak]:.3f} à {lags[peak]:+.3f} s",
        )

    def show_superimposed(self, ch4: ChannelData, ch3: ChannelData) -> None:
        def draw(ax):
            ax.plot(ch4.timestamps, ch4.samples, label=f"MDF4 - {ch4.name}")
            ax.plot(ch3.timestamps, ch3.samples, label=f"MDF3 - {ch3.name}", alpha=0.8)
            ax.legend(fontsize=8)

        units = {u for u in (ch4.unit, ch3.unit) if u}
        self.overlay_plot.show(
            "Superposition MDF4 & MDF3", "Temps (s)", " / ".join(sorted(units)) or "Amplitude", draw
        )

    def compute_pearson(self) -> None:
        if not self.check_ready(self.mdf4, self.mdf3):
            return
        fetch = self._fetch_both()

        def compute():
            ch4, ch3 = fetch()
            return analysis.pearson_correlation(ch4.timestamps, ch4.samples, ch3.timestamps, ch3.samples)

        self.result_var.set("")
        self.run_task(
            "Calcul du coefficient de Pearson", compute, lambda r: self.result_var.set(r.describe())
        )

    def plot_cross_correlation(self) -> None:
        if not self.check_ready(self.mdf4, self.mdf3):
            return
        fetch = self._fetch_both()

        def compute():
            ch4, ch3 = fetch()
            return analysis.cross_correlation(ch4.timestamps, ch4.samples, ch3.timestamps, ch3.samples)

        self.run_task("Calcul de la cross-corrélation", compute, lambda r: self.show_cross_correlation(*r))

    def plot_superimposed(self) -> None:
        if self.check_ready(self.mdf4, self.mdf3):
            self.run_task("Lecture des voies", self._fetch_both(), lambda r: self.show_superimposed(*r))

    def run_all(self) -> None:
        """Calcule toutes les analyses en une fois. Une analyse qui échoue
        (ex. signal constant) n'empêche pas l'affichage des autres."""
        if not self.check_ready(self.mdf4, self.mdf3):
            return
        fetch = self._fetch_both()

        def compute():
            ch4, ch3 = fetch()
            results, errors = {}, []
            jobs = {
                "auto4": (
                    "Auto-corrélation MDF4",
                    analysis.auto_correlation,
                    (ch4.timestamps, ch4.samples),
                ),
                "auto3": (
                    "Auto-corrélation MDF3",
                    analysis.auto_correlation,
                    (ch3.timestamps, ch3.samples),
                ),
                "xcorr": (
                    "Cross-corrélation",
                    analysis.cross_correlation,
                    (ch4.timestamps, ch4.samples, ch3.timestamps, ch3.samples),
                ),
                "pearson": (
                    "Coefficient de Pearson",
                    analysis.pearson_correlation,
                    (ch4.timestamps, ch4.samples, ch3.timestamps, ch3.samples),
                ),
            }
            for key, (label, func, args) in jobs.items():
                try:
                    results[key] = func(*args)
                except analysis.AnalysisError as exc:
                    errors.append(f"• {label} : {exc}")
            return ch4, ch3, results, errors

        def done(result):
            ch4, ch3, results, errors = result
            self.mdf4.show_channel(ch4)
            self.mdf3.show_channel(ch3)
            self.show_superimposed(ch4, ch3)
            if "auto4" in results:
                self.mdf4.show_auto_correlation(ch4, *results["auto4"])
            if "auto3" in results:
                self.mdf3.show_auto_correlation(ch3, *results["auto3"])
            if "xcorr" in results:
                self.show_cross_correlation(*results["xcorr"])
            self.result_var.set(results["pearson"].describe() if "pearson" in results else "")
            if errors:
                messagebox.showwarning(
                    "Analyse partielle",
                    "Certaines analyses n'ont pas pu être calculées :\n\n" + "\n".join(errors),
                    parent=self.root,
                )

        self.run_task("Analyse complète", compute, done)

    def reset(self) -> None:
        self.mdf4.clear()
        self.mdf3.clear()
        self.xcorr_plot.clear()
        self.overlay_plot.clear()
        self.result_var.set("")

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        for panel in (self.mdf4, self.mdf3):
            if panel.source is not None:
                panel.source.close()
        self.root.quit()
        self.root.destroy()


def create_root() -> tk.Tk:
    """Fenêtre principale, avec le thème « aquativo » si ttkthemes est installé."""
    try:
        from ttkthemes import ThemedTk

        return ThemedTk(theme="aquativo")
    except ImportError:
        return tk.Tk()


def _setup_logging() -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    if sys.stderr is None:
        # Exécutable Windows sans console : journal dans le dossier temporaire
        log_file = Path(tempfile.gettempdir()) / "MDF_ToolBox.log"
        logging.basicConfig(level=logging.INFO, format=fmt, filename=log_file, encoding="utf-8")
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)


def main() -> None:
    if "--self-test" in sys.argv[1:]:
        from .selftest import run

        sys.exit(run())
    _setup_logging()
    root = create_root()
    App(root)
    root.mainloop()
