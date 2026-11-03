"""Construit l'exécutable autonome de MDF ToolBox avec PyInstaller.

Usage (depuis la racine du dépôt) :
    pip install -r requirements.txt pyinstaller
    python scripts/build_exe.py

Résultat : dist/MDF_ToolBox.exe sous Windows (dist/MDF_ToolBox ailleurs).
Vérification : dist/MDF_ToolBox.exe --self-test
"""

from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent

PyInstaller.__main__.run(
    [
        str(ROOT / "main.py"),
        "--name=MDF_ToolBox",
        "--onefile",  # un seul fichier à distribuer
        "--windowed",  # pas de console noire derrière la fenêtre
        "--noconfirm",
        "--clean",
        f"--distpath={ROOT / 'dist'}",
        f"--workpath={ROOT / 'build'}",
        f"--specpath={ROOT / 'build'}",
        # Fichiers de thèmes Tk (thème « aquativo »)
        "--collect-data=ttkthemes",
        # Lecteurs MDF3/MDF4 chargés dynamiquement par asammdf
        "--collect-submodules=asammdf.blocks",
        # Nécessaire aux icônes de la barre d'outils Matplotlib
        "--hidden-import=PIL._tkinter_finder",
        # Interface Qt d'asammdf : inutile ici, et alourdirait l'exécutable
        "--exclude-module=asammdf.gui",
        "--exclude-module=PySide6",
        "--exclude-module=PyQt5",
        "--exclude-module=PyQt6",
    ]
)
