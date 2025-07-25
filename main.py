import os
import numpy as np
import pandas as pd

from loguru import logger

from mwe_db_access.config import settings
from mwe_db_access.db import (
    engine,
    get_saio_obj,
    register_schemas,
    session_scope,
)
from mwe_db_access.ssh import sshtunnel

# Parameter
N_ITERATIONS = 100  # Anzahl der Summen-Zeitreihen
SAMPLE_SIZE = 500  # Anzahl der zufälligen Profile pro Iteration
SEED = 42  # Seed für Reproduzierbarkeit
OUTPUT_SUM_PATH = "mwe_db_access/data/summenzeitreihen.csv"
OUTPUT_IDS_PATH = "ids_per_zeitreihe.csv"


def generate_summed_profiles(
    n_iterations: int = N_ITERATIONS,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
    output_sum: str = OUTPUT_SUM_PATH,
    output_ids: str = OUTPUT_IDS_PATH,
):
    """
    Baut einen SSH-Tunnel auf, lädt alle Profile-IDs,
    zieht per Zufall jeweils `sample_size` IDs, summiert
    deren 'load_in_wh'-Zeitreihen und speichert das Ergebnis.
    """
    rng = np.random.default_rng(seed)
    summen_list = []
    ids_list = []

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

        # 3) Iteratives Sampling & Summierung
        for i in range(1, n_iterations + 1):
            # 3a) Zufällige IDs auswählen
            sampled_ids = rng.choice(all_ids, size=sample_size, replace=False)
            ids_list.append(sampled_ids)

            # 3b) SQL-Query für diese IDs
            id_list_str = ",".join(map(str, sampled_ids))
            query = f"""
                SELECT id, load_in_wh
                FROM demand.iee_household_load_profiles
                WHERE id IN ({id_list_str})
                ORDER BY id
            """
            df = pd.read_sql(query, db_engine)

            # 3c) 'load_in_wh' Array-Spalte in Matrix verwandeln
            #     und entlang der Spalten summieren
            matrix = np.vstack(df["load_in_wh"].values)
            summed = matrix.sum(axis=0)
            summen_list.append(summed)

            logger.info(f"[{i}/{n_iterations}] Summen-Zeitreihe berechnet.")

    # 4) Ergebnisse in DataFrames umwandeln
    summen_df = pd.DataFrame(summen_list)
    ids_df = pd.DataFrame(ids_list)

    # 5) Speichern
    summen_df.to_csv(output_sum, index=False)
    ids_df.to_csv(output_ids, index=False)
    logger.info(f"Summen-Zeitreihen gespeichert in '{output_sum}'.")
    logger.info(f"IDs je Zeitreihe gespeichert in '{output_ids}'.")


if __name__ == "__main__":
    generate_summed_profiles()