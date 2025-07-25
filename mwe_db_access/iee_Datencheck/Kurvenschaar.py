import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loguru import logger

from mwe_db_access.config import settings
from mwe_db_access.db import engine, register_schemas
from mwe_db_access.ssh import sshtunnel

# Parameter
SAMPLE_SIZE = 500  # Anzahl der zufälligen Profile
SEED = None  # Seed für Reproduzierbarkeit
OUTPUT_IDS_PATH = "../data/sampled_ids.csv"


def plot_average_week_ensemble(
    sample_size: int = SAMPLE_SIZE,
    seed: int = None,
    output_ids: str = OUTPUT_IDS_PATH,
):
    """
    Baut einen SSH-Tunnel auf, lädt alle Profil-IDs,
    zieht per Zufall `sample_size` IDs, lädt die jeweiligen
    'load_in_wh'-Jahres-Zeitreihen, berechnet pro Profil
    die durchschnittliche Wochenlast (alle Wochen aufsummieren
    und dann durch Anzahl der Wochen teilen), und plottet
    diese durchschnittlichen Wochen als Kurvenschar.
    """
    rng = np.random.default_rng(seed)

    # 1) SSH-Tunnel & DB-Verbindung
    with sshtunnel(settings["ssh-tunnel"]["name"]):
        logger.info("SSH-Tunnel aufgebaut.")
        db_engine = engine()
        register_schemas(db_engine)

        # 2) Alle verfügbaren IDs holen (einmalig)
        logger.info("Lade alle Profil-IDs...")
        all_ids_df = pd.read_sql(
            "SELECT id FROM demand.iee_household_load_profiles", db_engine
        )
        all_ids = all_ids_df["id"].to_numpy()
        logger.info(f"{len(all_ids)} IDs geladen.")

        # 3) Einmaliges Sampling
        sampled_ids = rng.choice(all_ids, size=sample_size, replace=False)
        # IDs speichern (optional)
        ids_df = pd.DataFrame(sampled_ids, columns=["id"])
        ids_df.to_csv(output_ids, index=False)
        logger.info(f"{sample_size} zufällige IDs gespeichert in '{output_ids}'.")

        # 4) SQL-Query für die gesampelten IDs
        id_list_str = ",".join(map(str, sampled_ids))
        query = f"""
            SELECT id, load_in_wh
            FROM demand.iee_household_load_profiles
            WHERE id IN ({id_list_str})
            ORDER BY id
        """
        df = pd.read_sql(query, db_engine)
        logger.info("Zeitreihen für die gesampelten IDs geladen.")

    # 5) 'load_in_wh'-Arrays extrahieren und in Matrix umwandeln
    #    Jede Zeile der Matrix entspricht einer einzelnen Profil-Zeitreihe (für ein ganzes Jahr)
    matrix = np.vstack(df["load_in_wh"].values)  # shape: (sample_size, T)
    T = matrix.shape[1]
    logger.info(f"Jahreslänge der Zeitreihen: {T} Zeitpunkte.")

    # 6) Anzahl ganzer Wochen ermitteln (angenommen 1 Woche = 168 Stunden)
    hours_per_week = 168
    n_weeks = T // hours_per_week
    if n_weeks < 1:
        raise ValueError(f"Zeitreihe ist kürzer als eine Woche (Länge {T}).")
    trimmed_len = n_weeks * hours_per_week

    if trimmed_len < T:
        logger.warning(
            f"Länge {T} ist nicht exakt durch {hours_per_week} teilbar. "
            f"Schneide auf {trimmed_len} Punkte (erste {n_weeks} Wochen)."
        )

    # 7) Matrix zuschneiden und in (sample_size, n_weeks, 168) umformen
    matrix_trimmed = matrix[:, :trimmed_len]  # ignoriere überzählige Stunden
    matrix_weeks = matrix_trimmed.reshape(sample_size, n_weeks, hours_per_week)

    # 8) Durchschnittliche Woche pro Profil berechnen
    #    Summe über Achse 1 (alle Wochen) und dann geteilt durch n_weeks → shape: (sample_size, 168)
    avg_week_matrix = matrix_weeks.mean(axis=1)
    logger.info("Durchschnittliche Woche pro Profil berechnet.")

    # 9) Plotten der Kurvenschar der durchschnittlichen Wochen
    plt.figure(figsize=(12, 6))
    for week_profile in avg_week_matrix:
        plt.plot(np.arange(hours_per_week), week_profile, linewidth=0.8, alpha=0.3)
    plt.xlabel("Stunde der Woche (0–167)")
    plt.ylabel("Load in Wh")
    plt.title(f"Kurvenschaar: Durchschnittliche Woche von {sample_size} Haushalten")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Optional: Wochentags-Markierungen auf der x-Achse
    #   Jeden 24. Punkt (jede volle Tagesschleife) ein Label setzen
    ticks = np.arange(0, hours_per_week + 1, 24)
    labels = [
        "Sa",
        "So",
        "Mo",
        "Di",
        "Mi",
        "Do",
        "Fr",
        "Sa",
    ]  # +Mo am Ende, falls 168 Stunden
    plt.xticks(ticks, labels, rotation=0)

    plt.tight_layout()
    plt.show()
    logger.info("Kurvenschaar der durchschnittlichen Wochen geplottet.")


if __name__ == "__main__":
    plot_average_week_ensemble()
