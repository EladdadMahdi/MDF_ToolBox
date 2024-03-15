# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:40:02 2024
@author: p097220
Readme.md
Dev : P097220
DEASGC2
PDM

* French description need to be translated
Ce script Python est conçu pour fournir une\
interface utilisateur graphique (GUI) permettant l'analyse et\
la visualisation de données issues de fichiers de données de mesure (MDF),\
spécifiquement les versions MDF3 et MDF4.\
Il utilise customtkinter pour une apparence améliorée de l'interface utilisateur\
et matplotlib pour le traçage des données. Voici les fonctionnalités clés du script :\
Fonctionnalités Principales:\
Chargement des Fichiers MDF: Permet à l'utilisateur de charger des fichiers
MDF3 ou MDF4 à partir du système de fichiers en utilisant un dialogue de sélection de fichiers.
Affichage des Canaux: Après le chargement d'un fichier MDF, les canaux disponibles sont affichés
dans un menu déroulant, permettant à l'utilisateur de sélectionner spécifiquement quel canal visualiser.
Visualisation des Données: Offre la possibilité de tracer les données sélectionnées pour les canaux MDF3 et MDF4,
permettant une analyse visuelle des mesures enregistrées.
*Analyse de Signal:
Corrélation Croisée: Fonctionnalité pour calculer et tracer la corrélation croisée entre deux signaux sélectionnés,
idant à identifier les relations temporelles entre eux.
Auto-Correlation: Capacité à calculer et afficher l'auto-corrélation d'un signal, utile pour l'analyse des propriétés périodiques.
Corrélation de Pearson: Calcule et affiche le coefficient de corrélation de Pearson entre deux signaux,
fournissant une mesure de leur corrélation linéaire.
Superposition des Tracés: Permet la superposition des tracés des données MDF3 et MDF4 pour une comparaison directe,
facilitant l'identification des différences ou des anomalies entre les versions des fichiers.
*Structure du Script:
Le script organise son code en fonctions clairement définies pour chaque opération, comme le chargement des fichiers,
 la mise à jour des menus déroulants avec les canaux disponibles, et la visualisation des données.
Il intègre des pratiques de gestion d'erreur pour une meilleure robustesse, notamment lors du chargement des fichiers et de la visualisation des données.
La configuration de l'interface utilisateur et la logique métier sont séparées pour une meilleure clarté et maintenance du code.
*Utilisation:
Pour utiliser ce script, l'utilisateur doit exécuter le programme, qui lancera une interface graphique. À partir de là, il peut charger des fichiers MDF,
sélectionner des canaux pour l'analyse, et visualiser différents types de tracés et d'analyses de signal directement à partir de l'interface.
Ce script vise à fournir un outil complet pour l'analyse des fichiers MDF dans un environnement graphique convivial,
facilitant l'analyse des données de mesure automobile pour les ingénieurs et les techniciens.

"""
import io
import queue
import sys
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

# Importer la couleur rouge et la police en gras
from tkinter import ttk  # For a more modern style of dropdown list
from tkinter import filedialog, font, messagebox

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from asammdf import MDF
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from ttkthemes import ThemedTk

sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
sys.stderr = sys.stdout  # Redirect stderr to stdout


# Function to load MDF4 file using a file dialog
def load_mdf4():
    file_path = filedialog.askopenfilename(
        title="Select MDF4 File", filetypes=[("MDF files", "*.mf4")]
    )
    if file_path:
        global mdf4_file
        mdf4_file = MDF(file_path)
        update_channel_dropdown()


# Function to load MDF3 file using a file dialog
def load_mdf3():
    file_path_3 = filedialog.askopenfilename(
        title="Select MDF3 File", filetypes=[("MDF files", "*.mdf")]
    )
    if file_path_3:
        global mdf3_file_3
        mdf3_file_3 = MDF(file_path_3)
        update_channel_dropdown_3()


# Update the dropdown with channels from the loaded MDF4 file
def update_channel_dropdown():
    """scroll channels list MDF4"""
    if mdf4_file:
        channels = sorted(list(mdf4_file.channels_db))
        channel_var.set(channels[0])
        channel_dropdown["values"] = channels


# Update the dropdown with channels from the loaded MDF3 file
def update_channel_dropdown_3():
    """scroll channels list MDF3"""
    if mdf3_file_3:
        channels_3 = sorted(list(mdf3_file_3.channels_db))
        channel_var_3.set(channels_3[0])
        channel_dropdown_3["values"] = channels_3


# plot the dropdown with channels from the loaded MDF4 file
def plot_channel():
    global fig_mdf4, ax_mdf4, canvas_mdf4
    channel_name = channel_var.get()
    if channel_name and mdf4_file is not None:
        try:
            signal = mdf4_file.get(channel_name)
            ax_mdf4.clear()
            ax_mdf4.plot(signal.timestamps, signal.samples)
            ax_mdf4.set_title(f"MDF4 - {channel_name}", fontsize=11, fontname="Arial")
            ax_mdf4.set_xlabel("Time (s)", fontsize=11, fontname="Arial")
            ax_mdf4.set_ylabel(signal.unit, fontsize=11, fontname="Arial", rotation=90)
            # Annotate the plot with the number of points
            signal_time = signal.timestamps
            num_points = len(signal.samples)
            total_timestamps = signal_time[len(signal_time) - 1] - signal_time[0]
            natural_freq = num_points / total_timestamps
            # Combine both annotation texts
            combined_text = f"Nombre de points: {int(num_points)}\nFréquence: {int(natural_freq)} Hz"

            # Example annotation settings
            font_size = 10
            font_color = "red"
            background_color = "lightgrey"

            # Annotate the plot with adjusted visibility settings
            ax_mdf4.annotate(
                combined_text,
                xy=(0.01, 0.90),
                xycoords="axes fraction",
                ha="left",
                fontsize=font_size,
                fontname="Arial",
                color=font_color,
                bbox=dict(
                    facecolor=background_color,
                    edgecolor="black",
                    boxstyle="round,pad=0.1",
                ),
            )

            canvas_mdf4.draw()
        except Exception as e:
            print(f"Error in plotting MDF4 channel '{channel_name}': {e}")
    else:
        print("MDF4 file not loaded or channel not selected.")


# plot the dropdown with channels from the loaded MDF4 file
def plot_channel_3():
    global fig_mdf3, ax_mdf3, canvas_mdf3
    channel_name_3 = channel_var_3.get()
    if channel_name_3 and mdf3_file_3 is not None:
        try:
            signal_3 = mdf3_file_3.get(channel_name_3)
            ax_mdf3.clear()
            ax_mdf3.plot(signal_3.timestamps, signal_3.samples)
            ax_mdf3.set_title(f"MDF3 - {channel_name_3}", fontsize=11, fontname="Arial")
            ax_mdf3.set_xlabel("Time (s)", fontsize=11, fontname="Arial")
            ax_mdf3.set_ylabel(
                getattr(signal_3, "unit", "Unknown Unit"),
                fontsize=11,
                fontname="Arial",
                rotation=90,
            )
            # Annotate the plot with the number of points
            signal_time = signal_3.timestamps
            num_points = len(signal_3.samples)
            total_timestamps = signal_time[len(signal_time) - 1] - signal_time[0]
            natural_freq = num_points / total_timestamps
            combined_text = f"Nombre de points: {int(num_points)}\nFréquence: {int(natural_freq)} Hz"

            # Example annotation settings
            font_size = 10
            font_color = "red"
            background_color = "lightgrey"

            # Annotate the plot with adjusted visibility settings
            ax_mdf3.annotate(
                combined_text,
                xy=(0.01, 0.90),
                xycoords="axes fraction",
                ha="left",
                fontsize=font_size,
                fontname="Arial",
                color=font_color,
                bbox=dict(
                    facecolor=background_color,
                    edgecolor="black",
                    boxstyle="round,pad=0.1",
                ),
            )
            canvas_mdf3.draw()
        except Exception as e:
            print(f"Error in plotting MDF3 channel '{channel_name_3}': {e}")
    else:
        print("MDF3 file not loaded or channel not selected.")


def cross_correlate_signals(signal1, signal2, timestamps1, timestamps2):
    # Calculate the sample interval (assuming equally spaced samples and consistent sampling rate)
    encoder = OneHotEncoder()
    scaler = MinMaxScaler()
    # case both signal are numerical
    if np.issubdtype(signal1.dtype, np.number) and np.issubdtype(
        signal2.dtype, np.number
    ):  # Numeric signal
        if len(timestamps1) > 1 and len(timestamps2) > 1:
            sample_interval_1 = np.mean(np.diff(timestamps1))
            sample_interval_2 = np.mean(np.diff(timestamps2))
            sample_interval = (sample_interval_1 + sample_interval_2) / 2
        else:
            raise ValueError("Insufficient timestamp data to calculate sample interval")
        # normalize sample interval and handle zero denominator
        signal1_denominator = (
            np.mean(signal1)
            if np.mean(signal1) != 0 and np.isnan(np.mean(signal1))
            else 1
        )
        signal2_denominator = (
            np.mean(signal2)
            if np.mean(signal2) != 0 and np.isnan(np.mean(signal2))
            else 1
        )
        signal1_normalized = (
            (signal1 - signal1_denominator) / np.std(signal1)
            if np.std(signal1) != 0 and np.isnan(np.std(signal1))
            else signal1
        )
        signal2_normalized = (
            (signal2 - signal2_denominator) / np.std(signal2)
            if np.std(signal2) != 0 and np.isnan(np.std(signal2))
            else signal2
        )

        # Compute the cross-correlation
        correlation = scipy.signal.correlate(
            signal1_normalized,
            signal2_normalized,
            mode="full",
            method="fft",
        )
    # case signal1 numerical and signal2 non numerical
    if np.issubdtype(signal1.dtype, np.number) and not np.issubdtype(
        signal2.dtype, np.number
    ):
        if len(timestamps1) > 1 and len(timestamps2) > 1:
            sample_interval_1 = np.mean(np.diff(timestamps1))
            sample_interval_2 = np.mean(np.diff(timestamps2))
            sample_interval = (sample_interval_1 + sample_interval_2) / 2
        else:
            raise ValueError("Insufficient timestamp data to calculate sample interval")
        # normalize sample interval and handle zero denominator
        signal1_denominator = (
            np.mean(signal1)
            if np.mean(signal1) != 0 and np.isnan(np.mean(signal1))
            else 1
        )

        signal1_normalized = (
            (signal1 - signal1_denominator) / np.std(signal1)
            if np.std(signal1) != 0 and np.isnan(np.std(signal1))
            else signal1
        )

        signal2 = signal2.reshape(-1, 1)
        signal2_encoded = encoder.fit_transform(signal2)
        signal2_encoded = signal2_encoded.toarray()
        # Normalize using MinMaxScaler
        signal2_encoded_normalized = scaler.fit_transform(signal2_encoded)
        correlation = np.mean(
            [
                scipy.signal.correlate(
                    signal1_normalized,
                    signal2_encoded_normalized[:, i],
                    mode="full",
                    method="auto",
                )
                for i in range(signal2_encoded_normalized.shape[1])
            ],
            axis=0,
        )
        correlation = correlation.astype(np.float64)  # Ensure float data type
        correlation /= (
            np.max(correlation)
            if np.max(correlation) != 0 and not np.isnan(np.max(correlation))
            else 1
        )
    # case signal1 non numerical and signal2 numerical
    if not np.issubdtype(signal1.dtype, np.number) and np.issubdtype(
        signal2.dtype, np.number
    ):
        if len(timestamps1) > 1 and len(timestamps2) > 1:
            sample_interval_1 = np.mean(np.diff(timestamps1))
            sample_interval_2 = np.mean(np.diff(timestamps2))
            sample_interval = (sample_interval_1 + sample_interval_2) / 2
        else:
            raise ValueError("Insufficient timestamp data to calculate sample interval")
        # normalize sample interval and handle zero denominator
        signal2_denominator = (
            np.mean(signal2)
            if np.mean(signal2) != 0 and np.isnan(np.mean(signal2))
            else 1
        )

        signal2_normalized = (
            (signal2 - signal2_denominator) / np.std(signal2)
            if np.std(signal2) != 0 and np.isnan(np.std(signal2))
            else signal2
        )

        signal1 = signal1.reshape(-1, 1)
        signal1_encoded = encoder.fit_transform(signal1)
        signal1_encoded = signal1_encoded.toarray()
        # Normalize using MinMaxScaler
        signal1_encoded_normalized = scaler.fit_transform(signal1_encoded)
        correlation = np.mean(
            [
                scipy.signal.correlate(
                    signal2_normalized,
                    signal1_encoded_normalized[:, i],
                    mode="full",
                    method="auto",
                )
                for i in range(signal1_encoded_normalized.shape[1])
            ],
            axis=0,
        )
        correlation = correlation.astype(np.float64)  # Ensure float data type
        correlation /= (
            np.max(correlation)
            if np.max(correlation) != 0 and not np.isnan(np.max(correlation))
            else 1
        )
    # other cases
    if not np.issubdtype(signal1.dtype, np.number) and not np.issubdtype(
        signal2.dtype, np.number
    ):
        if len(timestamps1) > 1 and len(timestamps2) > 1:
            sample_interval_1 = np.mean(np.diff(timestamps1))
            sample_interval_2 = np.mean(np.diff(timestamps2))
            sample_interval = (sample_interval_1 + sample_interval_2) / 2
        else:
            raise ValueError("Insufficient timestamp data to calculate sample interval")
        # normalize sample interval and handle zero denominator

        # Signal1
        signal1 = signal1.reshape(-1, 1)
        signal1_encoded = encoder.fit_transform(signal1)
        signal1_encoded = signal1_encoded.toarray()

        # Signal2
        signal2 = signal2.reshape(-1, 1)
        signal2_encoded = encoder.fit_transform(signal2)
        signal2_encoded = signal2_encoded.toarray()

        correlation = np.mean(
            [
                scipy.signal.correlate(
                    signal1_encoded[:, i],
                    signal2_encoded[:, j],
                    mode="full",
                    method="auto",
                )
                for i in range(signal1_encoded.shape[1])
                for j in range(signal2_encoded.shape[1])
            ],
            axis=0,
        )
        correlation = correlation.astype(np.float64)  # Ensure float data type
        correlation /= (
            np.max(correlation)
            if np.max(correlation) != 0 and not np.isnan(np.max(correlation))
            else 1
        )

    # Calculate the number of lags
    num_lags = len(correlation)
    max_lag = (num_lags // 2) * sample_interval
    lags = np.linspace(-max_lag, max_lag, num_lags)
    return lags, correlation


# Function to plot the cross-correlation between the selected MDF3 and MDF4 channel with vertical bar
def plot_cross_correlation():
    global fig_corr, ax_corr, canvas_corr
    # MDF4
    signal_data_mdf4 = mdf4_file.get(channel_var.get())
    signal_mdf4 = signal_data_mdf4.samples
    timestamps_mdf4 = signal_data_mdf4.timestamps
    # MDF3
    signal_data_mdf3 = mdf3_file_3.get(channel_var_3.get())
    signal_mdf3 = signal_data_mdf3.samples
    timestamps_mdf3 = signal_data_mdf3.timestamps
    # Calculate cross-correlation and lags
    lags, correlation = cross_correlate_signals(
        signal_mdf4,
        signal_mdf3,
        timestamps_mdf4,
        timestamps_mdf3,
    )
    # Plotting
    ax_corr.clear()
    ax_corr.plot(lags, correlation)
    # Add a vertical line at x=0
    ax_corr.axvline(x=0, color="red", linestyle="--")
    ax_corr.set_title("Cross-Correlation MDF4 vs MDF3", fontsize=11, fontname="Arial")
    ax_corr.set_xlabel("Lag (seconds)", fontsize=11, fontname="Arial")
    ax_corr.set_ylabel("Cross-Correlation", fontsize=11, fontname="Arial", rotation=90)
    canvas_corr.draw_idle()
    return None, None


# ----------------------------------------------------------------
# test_ok
# ----------------------------------------------------------------
def auto_correlate_signal(signal, sample_interval):
    signal = np.nan_to_num(signal)
    encoder = OneHotEncoder()
    if np.issubdtype(signal.dtype, np.number):  # Numeric signal
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        if std_signal != 0:  # Ensure standard deviation is not zero
            signal_normalized = (signal - mean_signal) / std_signal
        else:
            signal_normalized = np.zeros_like(signal)  # If std is zero, set to zero
        auto_correlation = scipy.signal.correlate(
            signal_normalized, signal_normalized, mode="full", method="fft"
        )
        auto_correlation = auto_correlation.astype(np.float64)  # Ensure float data type
        auto_correlation /= (
            np.max(auto_correlation)
            if np.max(auto_correlation) != 0 and not np.isnan(np.max(auto_correlation))
            else 1
        )

    else:  # Non-numeric signal
        signal = signal.reshape(-1, 1)
        signal_encoded = encoder.fit_transform(signal)
        signal_encoded = signal_encoded.toarray()
        auto_correlation = np.mean(
            [
                scipy.signal.correlate(
                    signal_encoded[:, i],
                    signal_encoded[:, i],
                    mode="full",
                    method="auto",
                )
                for i in range(signal_encoded.shape[1])
            ],
            axis=0,
        )
        auto_correlation = auto_correlation.astype(np.float64)  # Ensure float data type
        auto_correlation /= (
            np.max(auto_correlation)
            if np.max(auto_correlation) != 0 and not np.isnan(np.max(auto_correlation))
            else 1
        )
    num_samples = len(signal)
    max_lag_seconds = (num_samples - 1) * sample_interval
    lags = np.linspace(0, max_lag_seconds, num_samples)
    return lags, auto_correlation[len(signal) - 1 :]


# # Function to calulcate sample interval
def get_sample_interval(timestamps):
    if len(timestamps) > 1:
        return np.mean(np.diff(timestamps))
    else:
        raise ValueError("Insufficient data to calculate sample interval")


# Function to plot the auto correlation of MDF3
def plot_auto_correlation_mdf4():
    global fig_auto_corr_mdf4, ax_auto_corr_mdf4, canvas_auto_corr_mdf4
    channel_name = channel_var.get()

    if mdf4_file:
        if channel_name:
            signal_data = mdf4_file.get(channel_name)
            signal = signal_data.samples
            timestamps = signal_data.timestamps
            sample_interval = get_sample_interval(timestamps)
            lags, auto_corr = auto_correlate_signal(signal, sample_interval)
            ax_auto_corr_mdf4.clear()
            ax_auto_corr_mdf4.plot(lags, auto_corr)
            ax_auto_corr_mdf4.set_title(
                f"MDF4 Auto-Correlation - {channel_name}", fontsize=11, fontname="Arial"
            )
            ax_auto_corr_mdf4.set_xlabel("Lag (seconds)", fontsize=11, fontname="Arial")
            ax_auto_corr_mdf4.set_ylabel(
                "Autocorrelation", fontsize=11, fontname="Arial", rotation=90
            )
            canvas_auto_corr_mdf4.draw()

        else:
            # If signal_mdf4_unit is empty, show a warning message
            messagebox.showwarning("Erreur: Pour continuer veuillez choisir une voie.")

    else:
        # If signal_mdf4_unit is empty, show a warning message
        messagebox.showwarning("Erreur: Pour continuer veuillez choisir un fichier.")


# Function to plot the auto correlation of MDF3
def plot_auto_correlation_mdf3():
    global fig_auto_corr_mdf3, ax_auto_corr_mdf3, canvas_auto_corr_mdf3
    channel_name_3 = channel_var_3.get()

    if mdf3_file_3:
        if channel_name_3:
            signal_data_3 = mdf3_file_3.get(channel_name_3)
            signal_3 = signal_data_3.samples
            timestamps_3 = signal_data_3.timestamps
            sample_interval_3 = get_sample_interval(timestamps_3)
            lags_3, auto_corr_3 = auto_correlate_signal(signal_3, sample_interval_3)
            ax_auto_corr_mdf3.clear()
            ax_auto_corr_mdf3.plot(lags_3, auto_corr_3)
            ax_auto_corr_mdf3.set_title(
                f"MDF3 Auto-Correlation - {channel_name_3}",
                fontsize=11,
                fontname="Arial",
            )
            ax_auto_corr_mdf3.set_xlabel("Lag (seconds)", fontsize=11, fontname="Arial")
            ax_auto_corr_mdf3.set_ylabel(
                "Autocorrelation", fontsize=11, fontname="Arial", rotation=90
            )
            canvas_auto_corr_mdf3.draw()
        else:
            # If signal_mdf4_unit is empty, show a warning message
            messagebox.showwarning("Erreur: Pour continuer veuillez choisir une voie.")
    else:
        # If signal_mdf4_unit is empty, show a warning message
        messagebox.showwarning("Erreur: Pour continuer veuillez choisir un fichier.")


#     return pearson_correlation
def calculate_pearson_correlation(signal1, signal2):
    # Create an instance of OrdinalEncoder
    encoder = OrdinalEncoder()

    # Ensuring the signals are of the same length
    min_len = min(len(signal1), len(signal2))
    signal1, signal2 = signal1[:min_len], signal2[:min_len]

    # Case: Both signals are numeric
    if np.issubdtype(signal1.dtype, np.number) and np.issubdtype(
        signal2.dtype, np.number
    ):
        correlation_matrix = np.corrcoef(signal1, signal2)
        return correlation_matrix[0, 1]

    # Case: signal1 is numeric and signal2 is non-numeric
    if np.issubdtype(signal1.dtype, np.number) and not np.issubdtype(
        signal2.dtype, np.number
    ):
        signal2_encoded = encoder.fit_transform(signal2.reshape(-1, 1))
        return np.corrcoef(signal1, signal2_encoded.T)[0, 1]

    # Case: signal1 is non-numeric and signal2 is numeric
    if not np.issubdtype(signal1.dtype, np.number) and np.issubdtype(
        signal2.dtype, np.number
    ):
        signal1_encoded = encoder.fit_transform(signal1.reshape(-1, 1))
        return np.corrcoef(signal1_encoded.T, signal2)[0, 1]

    # Case: Both signals are non-numeric
    if not np.issubdtype(signal1.dtype, np.number) and not np.issubdtype(
        signal2.dtype, np.number
    ):
        signal1_encoded = encoder.fit_transform(signal1.reshape(-1, 1))
        signal2_encoded = encoder.fit_transform(signal2.reshape(-1, 1))
        return np.corrcoef(signal1_encoded.T, signal2_encoded.T)[0, 1]

    # return np.nan  # Default case: Return NaN if none of the above conditions a


# Create a Text widget for displaying Pearson correlation
def plot_pearson_correlation():
    # try:
    #     # Assuming mdf4_file, channel_var, mdf3_file_3, and channel_var_3 are defined elsewhere
    #     signal_mdf4 = mdf4_file.get(channel_var.get()).samples
    #     signal_mdf3 = mdf3_file_3.get(channel_var_3.get()).samples

    #     # Calculate Pearson correlation using the calculate_pearson_correlation function
    #     pearson = calculate_pearson_correlation(signal_mdf4, signal_mdf3)

    #     # Display Pearson correlation in the Text widget (assuming text_box is defined elsewhere)
    #     text_box.delete("1.0", tk.END)  # Clear existing content
    #     # Créer une police en gras
    #     bold_font = font.Font(weight="bold")
    #     # Insérer le texte en gras et en rouge dans text_box
    #     text_box.tag_configure("bold_red", font=bold_font, foreground="red")
    #     text_box.insert(tk.END, f"Pearson correlation : {pearson}", "bold_red")

    # except Exception as e:
    #     # Handle any exceptions or errors
    #     text_box.delete("1.0", tk.END)  # Clear existing content
    #     text_box.insert(tk.END, f"Error: {str(e)}")

    # # Debugging output (can be commented out in production)
    # print(f"Pearson correlation coefficient: {pearson}")

    try:
        font_size = 12  # Default font size
        signal_mdf4 = mdf4_file.get(channel_var.get()).samples
        signal_mdf3 = mdf3_file_3.get(channel_var_3.get()).samples

        # Calculate Pearson correlation using the calculate_pearson_correlation function
        pearson = calculate_pearson_correlation(signal_mdf4, signal_mdf3)
        # Limit the precision of Pearson correlation
        precision = 3
        pearson = round(pearson, precision)
        # Display Pearson correlation in the Text widget
        text_box.delete("1.0", tk.END)  # Clear existing content

        # Create a font with bold and specified font size
        custom_font = font.Font(text_box, text_box.cget("font"))
        custom_font.configure(weight="bold", size=font_size)
        text_box.tag_configure("custom_font", font=custom_font, foreground="red")

        # Insert the text with the specified tag into the Text widget
        text_box.insert(tk.END, f"Pearson correlation : {pearson}\n", "custom_font")

    except Exception as e:
        # Handle any exceptions or errors
        text_box.delete("1.0", tk.END)  # Clear existing content
        text_box.insert(tk.END, f"Error: {str(e)}\n")

    # Debugging output (can be commented out in production)
    print(f"Pearson correlation coefficient: {pearson}")


# Function to update the overplot
def update_superimposed_plot():
    global fig_superimposed, ax_superimposed, canvas_superimposed

    # Retrieve your signals and their timestamps
    signal_data_mdf4 = mdf4_file.get(channel_var.get())
    signal_mdf4 = signal_data_mdf4.samples
    timestamps_mdf4 = signal_data_mdf4.timestamps
    signal_data_mdf3 = mdf3_file_3.get(channel_var_3.get())
    signal_mdf3 = signal_data_mdf3.samples
    timestamps_mdf3 = signal_data_mdf3.timestamps
    # Plot MDF4
    ax_superimposed.clear()
    ax_superimposed.plot(timestamps_mdf4, signal_mdf4, label="MDF4")
    # Plot MDF3 superimposed on the same plot
    ax_superimposed.plot(timestamps_mdf3, signal_mdf3, label="MDF3")
    # Set labels and legend
    ax_superimposed.set_title(
        "Superposition courbe MDF4 & MDF3", fontsize=11, fontname="Arial"
    )
    ax_superimposed.set_xlabel("Time (s)", fontsize=11, fontname="Arial")
    ax_superimposed.set_ylabel(
        "Signal Amplitude", fontsize=11, fontname="Arial", rotation=90
    )
    ax_superimposed.legend()
    # Draw the canvas
    canvas_superimposed.draw_idle()


# Function to plot
def perform_plot_operation(operation, *args, **kwargs):
    """
    Wraps plot operations to catch and handle exceptions.
    """
    try:
        operation(*args, **kwargs)
    except Exception as e:
        messagebox.showerror(
            "perform_plot_operation",
            f"Failed to perform operation {operation.__name__}: {e}",
        )


# Function to queue user ops
def worker_operation_queue(operation_queue):
    """
    Processes plot operations from a queue in a worker thread.
    """
    while not operation_queue.empty():
        operation, args, kwargs = operation_queue.get()
        perform_plot_operation(operation, *args, **kwargs)
        operation_queue.task_done()


# Function to run all operation by one click
def perform_all_operations():
    """
    Initiates plotting operations for both MDF3 and MDF4 files using threading to avoid GUI freezing.
    """
    signal1_unit = mdf4_file.get(channel_var.get()).unit
    signal2_unit = mdf3_file_3.get(channel_var_3.get()).unit

    if mdf3_file_3 and mdf4_file:
        # Create a queue for plot operations
        operation_queue = queue.Queue()

        signal1 = mdf4_file.get(channel_var.get()).samples if mdf4_file else None
        signal2 = mdf3_file_3.get(channel_var_3.get()).samples if mdf3_file_3 else None
        # MDF4
        signal = mdf4_file.get(channel_var.get()).samples if mdf4_file else None
        sample_interval = (
            get_sample_interval(mdf4_file.get(channel_var.get()).timestamps)
            if mdf4_file
            else None
        )
        timestamps = mdf4_file.get(channel_var.get()).timestamps if mdf4_file else None
        # MDF3
        signal = mdf3_file_3.get(channel_var_3.get()).samples if mdf3_file_3 else None
        sample_interval = (
            get_sample_interval(mdf3_file_3.get(channel_var_3.get()).timestamps)
            if mdf3_file_3
            else None
        )
        # Timestamps
        timestamps = (
            mdf3_file_3.get(channel_var_3.get()).timestamps if mdf3_file_3 else None
        )
        timestamps1 = mdf4_file.get(channel_var_3.get()).timestamps
        timestamps2 = mdf3_file_3.get(channel_var_3.get()).timestamps
        # Ensure signals are available before adding the operation
        if (
            signal is not None
            and signal1 is not None
            and signal2 is not None
            and sample_interval is not None
            and timestamps is not None
        ):
            # Queue up the plot operations
            operations = [
                (
                    cross_correlate_signals,
                    [
                        signal1,
                        signal2,
                        timestamps1,
                        timestamps2,
                    ],
                    {},
                ),
                (calculate_pearson_correlation, [signal1, signal2], {}),
                (auto_correlate_signal, [signal, sample_interval], {}),
                (get_sample_interval, [timestamps], {}),
                (plot_channel, [], {}),
                (plot_channel_3, [], {}),
                (plot_auto_correlation_mdf3, [], {}),
                (plot_auto_correlation_mdf4, [], {}),
                (plot_pearson_correlation, [], {}),
                (plot_cross_correlation, [], {}),
                (update_superimposed_plot, [], {}),
            ]

            def run_operations():
                # Define your operations here...
                operations = [
                    (
                        cross_correlate_signals,
                        [
                            signal1,
                            signal2,
                            timestamps1,
                            timestamps2,
                        ],
                        {},
                    ),
                    (calculate_pearson_correlation, [signal1, signal2], {}),
                    (auto_correlate_signal, [signal, sample_interval], {}),
                    (get_sample_interval, [timestamps], {}),
                    (plot_channel, [], {}),
                    (plot_channel_3, [], {}),
                    (plot_auto_correlation_mdf3, [], {}),
                    (plot_auto_correlation_mdf4, [], {}),
                    (plot_pearson_correlation, [], {}),
                    (plot_cross_correlation, [], {}),
                    (update_superimposed_plot, [], {}),
                ]

                # Perform operations and update progress
                total_operations = len(operations)
                for i, (operation, args, kwargs) in enumerate(operations, 1):
                    operation(*args, **kwargs)
                    progress_var.set(i / total_operations * 100)  # Update progress
                    root.update()

                # Close progress window when done
                progress_window.destroy()

            # Start a new thread to run the operations
            operation_thread = threading.Thread(target=run_operations)
            operation_thread.start()

            # Create progress window
            progress_window = tk.Toplevel(root)
            progress_window.title("Calcul en cours...")
            progress_window.geometry("520x50")
            progress_window.resizable(False, False)
            # Calculate the position to center the progress window
            window_width = progress_window.winfo_reqwidth()
            window_height = progress_window.winfo_reqheight()
            position_right = int(
                progress_window.winfo_screenwidth() / 2 - window_width / 2
            )
            position_down = int(
                progress_window.winfo_screenheight() / 2 - window_height / 2
            )

            # Position the progress window at the center of the GUI
            progress_window.geometry("+{}+{}".format(position_right, position_down))
            # Create progress bar
            progress_style = ttk.Style()
            progress_style.configure(
                "green.Horizontal.TProgressbar", foreground="green", thickness=10
            )

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_window,
                length=500,
                orient="horizontal",
                mode="determinate",
                variable=progress_var,
                style="green.Horizontal.TProgressbar",
            )
            progress_bar.pack(pady=20)
        else:
            messagebox.showwarning(
                "Avertissement", "Veuillez choisir des voies pour continuer."
            )
    else:
        messagebox.showinfo(
            "Information",
            "Veuillez sélectionner les deux fichiers MDF3 et MDF4 pour continuer.",
        )


def reset_all_operations():
    global fig_mdf4, ax_mdf4, canvas_mdf4, fig_mdf3, ax_mdf3, canvas_mdf3, text_box

    # Clear the plots
    if ax_mdf4:
        ax_mdf4.clear()
        canvas_mdf4.draw()
    if ax_mdf3:
        ax_mdf3.clear()
        canvas_mdf3.draw()
    if ax_corr:
        ax_corr.clear()
        canvas_corr.draw()
    if ax_superimposed:
        ax_superimposed.clear()
        canvas_superimposed.draw()
    if ax_auto_corr_mdf4:
        ax_auto_corr_mdf4.clear()
        canvas_auto_corr_mdf4.draw()
    if ax_auto_corr_mdf3:
        ax_auto_corr_mdf3.clear()
        canvas_auto_corr_mdf3.draw()

    # Clear the Pearson correlation text box
    if text_box:
        text_box.delete("1.0", tk.END)


# Désactive le mouvement de la souris sur tous les widgets de cross_corr_frame
def disable_mouse_motion(event):
    return "break"


######################################
#                 GUI                #
######################################
# Initialize the main window
root = ThemedTk(theme="aquativo")
root.title("Contrôle qualité conversion MDF4_MDF3 V.1.3")
# window_width = 1235
# window_height = 1010
# Fix the window for 17'
frame_width = 410
frame_height = 300
window_width = frame_width * 3 + 20  # Add padding
window_height = 1010
root.geometry(f"{window_width}x{window_height}")
# Prevent resizing of the window in the horizontal direction
root.resizable(False, False)
# # figure size
fig_width = 4
fig_height = 3.7
# Frames of the GUI
# Define the size of the frame
# #frame_width = 410
# frame_height = 300

# frame_width = 410
# window_width = frame_width * 3 + 20  # Add padding
# window_height = 1010
# MDF3 FRAME
mdf3_frame = tk.Frame(root, width=frame_width, height=frame_height, bg="lightblue")
mdf3_frame.grid_propagate(False)
mdf3_frame.bind("<Enter>", lambda event: None)  # Désactive le survol de la souris
mdf3_frame.bind("<Leave>", lambda event: None)
mdf3_frame.bind("<Enter>", lambda event: "break")  # Désactive le survol de la souris
mdf3_frame.bind("<Leave>", lambda event: "break")  # Désactive le départ de la souris
mdf3_frame.bind_all("<Motion>", lambda event: "break")

# MDF4 FRAME
mdf4_frame = tk.Frame(root, width=frame_width, height=frame_height, bg="lightblue")
mdf4_frame.grid_propagate(False)
mdf4_frame.bind("<Enter>", lambda event: None)  # Désactive le survol de la souris
mdf4_frame.bind("<Leave>", lambda event: None)  # Désactive le départ de la souris
mdf4_frame.bind("<Enter>", lambda event: "break")  # Désactive le survol de la souris
mdf4_frame.bind("<Leave>", lambda event: "break")  # Désactive le départ de la souris
mdf4_frame.bind_all("<Motion>", lambda event: "break")

# CROSS CORR FRAME
cross_corr_frame = tk.Frame(
    root, width=frame_width, height=frame_height, bg="lightblue"
)

# cross_corr_frame.grid_propagate(False)
cross_corr_frame.bind("<Enter>", lambda event: None)  # Désactive le survol de la souris
cross_corr_frame.bind("<Leave>", lambda event: None)  # Désactive le départ de la souris
cross_corr_frame.bind_all("<Motion>", lambda event: "break")
cross_corr_frame.bind(
    "<Enter>", lambda event: "break"
)  # Désactive le survol de la souris
cross_corr_frame.bind("<Leave>", lambda event: "break")

for widget in cross_corr_frame.winfo_children():
    widget.bind("<Enter>", lambda event: "break")  # Disable hover
    widget.bind("<Leave>", lambda event: "break")  # Disable leave
    widget.bind_all("<Motion>", lambda event: "break")  # Disable motion
    widget.bind("<Enter>", lambda event: None)  # Disable hover
    widget.bind("<Leave>", lambda event: None)

for widget in mdf3_frame.winfo_children():
    widget.bind("<Enter>", lambda event: "break")  # Disable hover
    widget.bind("<Leave>", lambda event: "break")  # Disable leave
    widget.bind("<Enter>", lambda event: None)  # Disable hover
    widget.bind("<Leave>", lambda event: None)  # Disable leave
    widget.bind_all("<Motion>", lambda event: "break")

    # Disable mot
for widget in mdf4_frame.winfo_children():
    widget.bind("<Enter>", lambda event: "break")  # Disable hover
    widget.bind("<Leave>", lambda event: "break")  # Disable leave
    widget.bind("<Enter>", lambda event: None)  # Disable hover
    widget.bind("<Leave>", lambda event: None)  # Disable leave
    widget.bind_all("<Motion>", lambda event: "break")  # Disable mot

# Gloabal variables
mdf3_file_3 = None
mdf4_file = None

# StringVar for channel dropdowns
channel_var = tk.StringVar()
channel_var_3 = tk.StringVar()

# Frames position in the GUI
# MDF4
mdf4_frame.grid(row=0, column=0, sticky="nsew")
# MDF3
mdf3_frame.grid(row=0, column=1, sticky="nsew")
# CROSS_CORR_FRAME
cross_corr_frame.grid(row=0, column=2, sticky="nsew")

# Exemple de taille minimale de 400 pixels
# Configure rows and columns of frames to expand
for frame in [mdf3_frame, mdf4_frame, cross_corr_frame]:
    # frame.grid_propagate(False)
    for row in range(10):  # Rows from 0 to 8
        frame.grid_rowconfigure(row, weight=1)
    for column in range(10):  # Columns from 0 to 2
        frame.grid_columnconfigure(column, weight=1)

# ----------------------------------------------------------------
# MDF4 Interface Elements
# ----------------------------------------------------------------
ttk.Button(mdf4_frame, text="Importer fichier MDF4", command=load_mdf4, width=40).grid(
    row=0, column=0, padx=5, pady=5, sticky="nsew"
)

channel_dropdown = ttk.Combobox(
    mdf4_frame, textvariable=channel_var, values=[], width=40
)

channel_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

ttk.Button(mdf4_frame, text="Tracer voie MDF4", command=plot_channel, width=40).grid(
    row=2, column=0, padx=5, pady=5, sticky="nsew"
)

fig_mdf4, ax_mdf4 = plt.subplots(figsize=(fig_width, fig_height))
canvas_mdf4 = FigureCanvasTkAgg(fig_mdf4, master=mdf4_frame)
widget_mdf4 = canvas_mdf4.get_tk_widget()
widget_mdf4.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

# For MDF4 canvas
toolbar_mdf4 = NavigationToolbar2Tk(canvas_mdf4, mdf4_frame, pack_toolbar=False)
toolbar_mdf4.update()
toolbar_mdf4.grid(row=5, column=0, padx=5, pady=5, sticky="nsew")

canvas_mdf4.get_tk_widget().grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

ttk.Button(
    mdf4_frame,
    text="Tracer Auto-Correlation MDF4",
    command=plot_auto_correlation_mdf4,
    width=40,
).grid(row=6, column=0, padx=5, pady=5, sticky="nsew")

fig_auto_corr_mdf4, ax_auto_corr_mdf4 = plt.subplots(figsize=(fig_width, fig_height))
canvas_auto_corr_mdf4 = FigureCanvasTkAgg(fig_auto_corr_mdf4, master=mdf4_frame)
widget_auto_corr_mdf4 = canvas_auto_corr_mdf4.get_tk_widget()
widget_auto_corr_mdf4.grid(row=7, column=0, padx=5, pady=5, sticky="nsew")

# For Auto Correlation MDF4 canvas
toolbar_auto_corr_mdf4 = NavigationToolbar2Tk(
    canvas_auto_corr_mdf4, mdf4_frame, pack_toolbar=False
)
toolbar_auto_corr_mdf4.update()
toolbar_auto_corr_mdf4.grid(row=8, column=0, padx=5, pady=5, sticky="nsew")

canvas_auto_corr_mdf4.get_tk_widget().grid(row=7, column=0, sticky="nsew")

# ----------------------------------------------------------------
# MDF3 Interface Elements
# ----------------------------------------------------------------
ttk.Button(mdf3_frame, text="Importer fichier MDF3", command=load_mdf3, width=40).grid(
    row=0, column=1, padx=5, pady=5, sticky="nsew"
)

channel_dropdown_3 = ttk.Combobox(
    mdf3_frame, textvariable=channel_var_3, values=[], width=40
)

channel_dropdown_3.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

ttk.Button(mdf3_frame, text="Tracer voie MDF3", command=plot_channel_3, width=40).grid(
    row=2, column=1, padx=5, pady=5, sticky="nsew"
)

fig_mdf3, ax_mdf3 = plt.subplots(figsize=(fig_width, fig_height))
canvas_mdf3 = FigureCanvasTkAgg(fig_mdf3, master=mdf3_frame)
widget_mdf3 = canvas_mdf3.get_tk_widget()
widget_mdf3.grid(row=4, column=1, padx=5, pady=5, sticky="nsew")

# # For MDF3 canvas
toolbar_mdf3 = NavigationToolbar2Tk(canvas_mdf3, mdf3_frame, pack_toolbar=False)
toolbar_mdf3.update()
toolbar_mdf3.grid(row=5, column=1, padx=5, pady=5, sticky="nsew")

canvas_mdf3.get_tk_widget().grid(row=4, column=1, padx=5, pady=5, sticky="nsew")

ttk.Button(
    mdf3_frame,
    text="Tracer Auto-Correlation MDF3",
    command=plot_auto_correlation_mdf3,
    width=40,
).grid(row=6, column=1, padx=5, pady=5, sticky="nsew")

fig_auto_corr_mdf3, ax_auto_corr_mdf3 = plt.subplots(figsize=(fig_width, fig_height))
canvas_auto_corr_mdf3 = FigureCanvasTkAgg(fig_auto_corr_mdf3, master=mdf3_frame)
widget_auto_corr_mdf3 = canvas_auto_corr_mdf3.get_tk_widget()
widget_auto_corr_mdf3.grid(row=7, column=1, padx=5, pady=5, sticky="nsew")

# For Auto Correlation MDF3 canvas
toolbar_auto_corr_mdf3 = NavigationToolbar2Tk(
    canvas_auto_corr_mdf3, mdf3_frame, pack_toolbar=False
)
toolbar_auto_corr_mdf3.update()
toolbar_auto_corr_mdf3.grid(row=8, column=1, padx=5, pady=5, sticky="nsew")

canvas_auto_corr_mdf3.get_tk_widget().grid(row=7, column=1, sticky="nsew")

# ----------------------------------------------------------------
# Correlation Buttons and Plot
# ----------------------------------------------------------------
ttk.Button(
    cross_corr_frame,
    text="Calcul coefficient de Pearson",
    command=plot_pearson_correlation,
    width=40,
).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

text_box = tk.Text(cross_corr_frame, height=1, width=40)

text_box.grid(row=1, column=0, padx=3, pady=3, sticky="nsew")

ttk.Button(
    cross_corr_frame, text="Lancer Correlation", command=perform_all_operations
).grid(row=2, column=0, padx=5, pady=5, sticky="nw")


ttk.Button(
    cross_corr_frame, text="Tracer Cross-Correlation", command=plot_cross_correlation
).grid(row=2, column=0, padx=5, pady=5, sticky="n")

# ----------------------------------------------------------------
# Reset button
# ----------------------------------------------------------------

ttk.Button(
    cross_corr_frame, text="Réinitialisation", command=reset_all_operations
).grid(row=2, column=0, padx=5, pady=5, sticky="ne")

fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
canvas_corr = FigureCanvasTkAgg(fig_corr, master=cross_corr_frame)
widget_corr = canvas_corr.get_tk_widget()
widget_corr.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

# For cross-correlation canvas
toolbar_corr = NavigationToolbar2Tk(canvas_corr, cross_corr_frame, pack_toolbar=False)
toolbar_corr.update()
toolbar_corr.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

canvas_corr.get_tk_widget().grid(row=3, column=0, sticky="nsew")

# ----------------------------------------------------------------
# Superposition
# ----------------------------------------------------------------
ttk.Button(
    cross_corr_frame,
    text="Superposition MDF3 et MDF4",
    command=update_superimposed_plot,
    width=40,
).grid(row=5, column=0, padx=5, pady=5, sticky="nsew")

fig_superimposed, ax_superimposed = plt.subplots(figsize=(fig_width, fig_height))
canvas_superimposed = FigureCanvasTkAgg(fig_superimposed, master=cross_corr_frame)
# Désactiver les événements de la souris pour le widget du graphique
canvas_superimposed.get_tk_widget().bind("<Motion>", lambda event: "break")
canvas_superimposed.get_tk_widget().bind(
    "<Enter>", lambda event: None
)  # Désactiver le survol
canvas_superimposed.get_tk_widget().bind("<Leave>", lambda event: None)
# Désactiver le départ
canvas_superimposed.get_tk_widget().bind("<Motion>", lambda event: "break")
canvas_superimposed.get_tk_widget().bind(
    "<Enter>", lambda event: "break"
)  # Désactiver le survol
canvas_superimposed.get_tk_widget().bind(
    "<Leave>", lambda event: "break"
)  # Désactiver le départ

widget_superimposed = canvas_superimposed.get_tk_widget()
widget_superimposed.grid(row=6, column=0, padx=5, pady=5, sticky="nsew")

# Update idletasks to ensure proper configuration
root.update()
# For Superimposed canvas
toolbar_superimposed = NavigationToolbar2Tk(
    canvas_superimposed, cross_corr_frame, pack_toolbar=False
)
toolbar_superimposed.update()
toolbar_superimposed.grid(row=7, column=0, padx=5, pady=5, sticky="nsew")
canvas_superimposed.get_tk_widget().grid(row=6, column=0, sticky="nsew")
# Disable mouse motion for the superimposed canvas
for widget in canvas_superimposed.get_tk_widget().winfo_children():
    widget.bind("<Motion>", lambda event: "break")
    widget.bind("<Enter>", lambda event: "break")  # Disable hover
    widget.bind("<Leave>", lambda event: "break")  # Disable leave

# Bind mouse motion event to the root window
root.bind("<Motion>", lambda event: "break")
root.bind("<Enter>", lambda event: "break")  # Disable hover
root.bind("<Leave>", lambda event: "break")  # Disable leave

# ----------------------------------------------------------------
# start gui
# ----------------------------------------------------------------
root.mainloop()
