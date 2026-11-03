# MDF File Channel Visualization Tool

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![asammdf](https://img.shields.io/badge/asammdf-%E2%89%A57.4.5-green)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-red)
![CI](https://github.com/EladdadMahdi/MDF_ToolBox/actions/workflows/ci.yml/badge.svg)

## Description

_The GUI is in French. An English translation is planned._

A desktop tool (Tkinter + Matplotlib) for comparing a channel recorded in two
**MDF** measurement files (MDF3 and/or MDF4), for example to validate that a
new acquisition chain records the same signal as the old one.

## Features

### File management
- Load MDF3 or MDF4 files (`.mdf`, `.mf4`, `.dat`). The version is detected from the file itself.
- Channel list with **type-to-filter** search.
- Files are loaded in the background: the window stays responsive.

### Visualization
- Plot a channel with its number of points and sampling frequency.
- Overlay the two channels on the same time axis.
- Zoom, pan and save the image from each plot's toolbar.

### Signal analysis
All comparisons first **resample both signals onto a common time base**,
restricted to the time range covered by both. This makes the results correct
even when the two files have different sampling rates or start times.

| Analysis | Numeric signals | Text / state signals |
|---|---|---|
| **Cross-correlation** | Normalised (in [-1, 1]); the peak gives the time delay between the two files | Averaged over the states the two signals share |
| **Auto-correlation** | Normalised to 1 at zero lag | Averaged over the states (one-hot encoding) |
| **Similarity** | Pearson correlation coefficient | Percentage of time where both states are equal |

A positive cross-correlation lag means the MDF4 signal is **late** relative to the MDF3 signal.

"Analyse complète" runs every analysis in one click. If one analysis cannot be
computed (for example, a constant signal), the others are still displayed, and
a message explains what failed.

## Project structure

```
app/
├── analysis.py   # Signal processing: pure functions, no GUI (tested)
├── mdf_io.py     # MDF file reading (asammdf) with a channel cache
├── gui.py        # Tkinter interface; computations run in a worker thread
└── __main__.py   # `python -m app`
main.py           # `python main.py`
tests/            # pytest: analysis, real MDF3/MDF4 files, GUI smoke tests
```

## Installation

Requires Python 3.10 or later, with Tkinter (included with Python on Windows
and macOS; on Debian/Ubuntu: `sudo apt install python3-tk`).

```bash
pip install -r requirements.txt
```

`ttkthemes` is optional. Without it, the default Tk theme is used.

## Usage

```bash
python main.py        # or: python -m app
```

1. Import an MDF4 file and an MDF3 file.
2. Choose a channel in each file (type to filter the list).
3. Plot each channel and its auto-correlation, or click **Analyse complète**.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest                      # GUI tests are skipped when there is no display
xvfb-run -a pytest          # Linux, headless, including GUI tests
```

Continuous integration (GitHub Actions) runs these checks on every push.

## Author
**Eladdad Mahdi**
