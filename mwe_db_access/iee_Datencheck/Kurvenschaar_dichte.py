import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loguru import logger

from mwe_db_access.config import settings
from mwe_db_access.db import engine, register_schemas
from mwe_db_access.ssh import sshtunnel

# ------------------------------------------------------------------------
# Parameter
# ------------------------------------------------------------------------
SAMPLE_SIZE = 500  # Anzahl der zufälligen Profile
SEED = None  # Seed für Reproduzierbarkeit
OUTPUT_IDS_PATH = "sampled_ids.csv"  # Pfad zum Speichern der Sample-IDs


# ------------------------------------------------------------------------
# Funktion: Plot der durchschnittlichen Wochenprofile mit Dichte-Heatmap
# ------------------------------------------------------------------------
def plot_average_week_ensemble(
    sample_size: int = SAMPLE_SIZE,
    seed: int = None,
    output_ids: str = OUTPUT_IDS_PATH,
):
    """
    1) Aufbau eines SSH-Tunnels und Laden aller Profil-IDs
    2) Zufälliges Sampling von `sample_size` IDs
    3) Laden der zugehörigen Last-Zeitreihen (load_in_wh)
    4) Berechnung der durchschnittlichen Wochenlast pro Profil
    5) Darstellung der Kurvenschar:
       a) Dichte-Heatmap mit PowerNorm und Konturen
       b) Optional: Plot aller Einzelkurven als Linien
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------
    # 1) SSH-Tunnel & DB-Verbindung
    # --------------------------------------------------------------------
    with sshtunnel(settings["ssh-tunnel"]["name"]):
        logger.info("SSH-Tunnel aufgebaut.")
        db_engine = engine()
        register_schemas(db_engine)

        # ----------------------------------------------------------------
        # 2) Alle Profil-IDs abrufen
        # ----------------------------------------------------------------
        logger.info("Lade alle Profil-IDs...")
        all_ids_df = pd.read_sql(
            "SELECT id FROM demand.iee_household_load_profiles", db_engine
        )
        all_ids = all_ids_df["id"].to_numpy()
        logger.info(f"{len(all_ids)} Profile gefunden.")

        # ----------------------------------------------------------------
        # 3) Zufälliges Sampling
        # ----------------------------------------------------------------
        sampled_ids = rng.choice(all_ids, size=sample_size, replace=False)
        pd.DataFrame(sampled_ids, columns=["id"]).to_csv(output_ids, index=False)
        logger.info(f"{sample_size} zufällige IDs gespeichert in '{output_ids}'.")

        # ----------------------------------------------------------------
        # 4) Laden der Zeitreihen für die gesampelten IDs
        # ----------------------------------------------------------------
        id_list_str = ",".join(map(str, sampled_ids))
        query = f"""
            SELECT id, load_in_wh
            FROM demand.iee_household_load_profiles
            WHERE id IN ({id_list_str})
            ORDER BY id
        """
        df = pd.read_sql(query, db_engine)
        logger.info("Zeitreihen geladen.")

    # ------------------------------------------------------------------------
    # 5) Aufbau der Datenmatrix
    # ------------------------------------------------------------------------
    # Jede Zeile = ein Haushalt, jede Spalte = ein Zeitstempel (Stunde)
    matrix = np.vstack(df["load_in_wh"].values)  # shape: (sample_size, T)
    T = matrix.shape[1]
    logger.info(f"Jahreslänge jeder Zeitreihe: {T} Stunden.")

    # Vollständige Wochen extrahieren (168 h/Woche)
    hours_per_week = 168
    n_weeks = T // hours_per_week
    if n_weeks < 1:
        raise ValueError("Zeitreihe kürzer als eine Woche!")
    trimmed_len = n_weeks * hours_per_week
    if trimmed_len < T:
        logger.warning(
            f"Schneide Zeitreihen von {T} auf {trimmed_len} Stunden "
            f"(erste {n_weeks} Wochen)."
        )
    matrix_trimmed = matrix[:, :trimmed_len]
    matrix_weeks = matrix_trimmed.reshape(sample_size, n_weeks, hours_per_week)

    # Durchschnittliche Woche pro Haushalt
    avg_week_matrix = matrix_weeks.mean(axis=1)
    logger.info("Durchschnittliche Woche pro Profil berechnet.")

    # ------------------------------------------------------------------------
    # 6a) Dichte-Heatmap mit PowerNorm und Konturen
    # ------------------------------------------------------------------------
    # Punktewolke: x = Stunde der Woche, y = Last-Wert
    hours = np.arange(hours_per_week)  # Werte 0–167
    x = np.tile(hours, sample_size)  # Wiederholung für jedes Profil
    y = avg_week_matrix.flatten()  # Alle Last-Werte in Serie

    # 2D-Histogramm berechnen (Feingitter in y für bessere Auflösung)
    H, xedges, yedges = np.histogram2d(
        x, y, bins=[hours_per_week, 100]  # 168 Bins in x, 100 Bins in y
    )

    # Normierung: PowerNorm komprimiert hohe Counts (<200) und betont
    # mittlere Werte; clip=True schneidet über 200 ab
    norm = mpl.colors.PowerNorm(gamma=0.4, vmin=0, vmax=200, clip=True)
    cmap = plt.cm.plasma

    plt.figure(figsize=(12, 6))
    # pcolormesh ermöglicht Konturen-Overlay
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H.T, cmap=cmap, norm=norm)

    # Konturlinien bei ausgewählten Schwellen
    levels = [50, 100, 150, 200]
    CS = plt.contour(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2,
        H.T,
        levels=levels,
        colors="white",
        linewidths=1,
    )
    plt.clabel(CS, fmt="%d", inline=1, fontsize=8, colors="white")

    # Achsen & Titel
    plt.xlabel("Stunde der Woche (0–167)")
    plt.ylabel("Load in Wh")
    plt.title(
        f"Dichte-Heatmap (PowerNorm γ=0.4) mit Konturen\n"
        f"{sample_size} Haushalte, erste {n_weeks} Wochen"
    )
    cb = plt.colorbar(label="Anzahl Profile (γ-skaliert)")
    cb.set_ticks([0, 50, 100, 150, 200])
    cb.set_ticklabels(["0", "50", "100", "150", "≥200"])

    # Wochentagsbeschriftung auf der x-Achse
    ticks = np.arange(0, hours_per_week + 1, 24)
    labels = ["Sa", "So", "Mo", "Di", "Mi", "Do", "Fr", "Sa"]
    plt.xticks(ticks, labels)

    plt.tight_layout()
    plt.show()
    logger.info("Dichte-Heatmap geplottet.")

    # ------------------------------------------------------------------------
    # 6b) Optional: Plot aller Einzelkurven als Linien
    # ------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    for profile in avg_week_matrix:
        plt.plot(hours, profile, linewidth=0.8, alpha=0.3)
    plt.xlabel("Stunde der Woche (0–167)")
    plt.ylabel("Load in Wh")
    plt.title(f"Kurvenschaar: Durchschnittliche Woche von {sample_size} Haushalten")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(ticks, labels)
    plt.tight_layout()
    plt.show()
    logger.info("Einzelkurven geplottet.")


# ------------------------------------------------------------------------
# Script-Einstiegspunkt
# ------------------------------------------------------------------------
if __name__ == "__main__":
    plot_average_week_ensemble()
