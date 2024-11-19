
# MDF Analysis and Visualization Tool

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.8.2-blue)
![Pandas](https://img.shields.io/badge/Pandas-v2.2.0-150458)
![NumPy](https://img.shields.io/badge/NumPy-v1.26.3-orange)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-red)
![Asammdf](https://img.shields.io/badge/Asammdf-v7.4.1-green)
![CanMatrix](https://img.shields.io/badge/CanMatrix-v1.0-yellow)

## Description

_The current GUI is in French. A translation to English is planned for a future update. Stay tuned!_

This Python script provides a **Graphical User Interface (GUI)** for analyzing and visualizing data from **Measurement Data Format (MDF)** files, specifically **MDF3** and **MDF4** versions. It is built with `customtkinter` for an enhanced user experience and `matplotlib` for data plotting.

The tool is designed to simplify the analysis of automotive measurement data, making it accessible to engineers and technicians in a user-friendly graphical environment. 

---

## Features

### **1. File Management**
- Load MDF3 or MDF4 files using a file selection dialog.
- Display available channels from the loaded file in a dropdown menu.

### **2. Data Visualization**
- Plot data from selected MDF3 or MDF4 channels for visual analysis.
- Overlay plots for MDF3 and MDF4 data for comparison.

### **3. Signal Analysis**
- **Cross-Correlation:** Calculate and visualize temporal relationships between two signals.
- **Auto-Correlation:** Analyze periodic properties of a signal.
- **Pearson Correlation:** Compute the Pearson correlation coefficient between two signals to assess their linear relationship.

---

## Script Structure

- **Modular Design:** Code is divided into functions for tasks like file loading, updating channel menus, and data visualization.
- **Error Handling:** Includes mechanisms to handle errors during file operations and visualization tasks.
- **Separation of Concerns:** User interface configuration and business logic are separated for improved maintainability.

---

## Installation

### **Requirements**
The script uses the following libraries:
- `customtkinter`
- `matplotlib`
- `numpy`
- `pandas`

Install the required dependencies using:
```bash
pip install -r requirements.txt
