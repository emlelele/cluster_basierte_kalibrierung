#!/usr/bin/env python3
"""
Wendet die stündlichen, global-adjusted Skalierungsfaktoren αᵢᵃᵈʲ(t)
auf alle 100 000 Einzelprofile an, batch‐weise und speichert die kalibrierten
Profile in CSV‐Chunks.

Vorgehen:
1. Lädt:
   - Cluster-Labels (id → Cluster i)
   - Kalibrierungsmatrix αᵃᵈʲ (K×T)
2. Öffnet SSH‐Tunnel & DB‐Verbindung
3. Liest in Batches (z.B. 5000 IDs) die Original‐Profile (load_in_wh)
4. Multipliziert jedes Profil j in Cluster i mit αᵃᵈʲᵢ(t)
5. Schreibt pro Batch eine CSV mit id und kalibriertem Zeitreihen‐Array

Usage:
    python apply_calibration_to_profiles.py
"""
import json
import os

import numpy as np
import pandas as pd

from loguru import logger

from mwe_db_access.config import settings
from mwe_db_access.db import engine, register_schemas
from mwe_db_access.ssh import sshtunnel

# -------------- Pfade & Parameter --------------
LABELS_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_labels_full_ts.csv"
ALPHAS_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/calibration_matrix_bounded_adjusted.npy"
OUTPUT_DIR = "../../calibrated_profiles_batches"
BATCH_SIZE = 1000  # Anzahl Profile pro Batch

# DB‐Konfiguration für die Tabelle mit load_in_wh
TABLE_NAME = "demand.iee_household_load_profiles"
ID_COLUMN = "id"
TS_COLUMN = "load_in_wh"  # gespeicherte JSON‐Liste von Wh/Werten

# -------------- Vorbereitung --------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Labels und α laden
labels_df = pd.read_csv(LABELS_CSV)  # Spalten: id, cluster
id_array = labels_df[ID_COLUMN].to_numpy()
cluster_array = labels_df["cluster"].to_numpy()
id_to_cluster = dict(zip(id_array, cluster_array))

alphas_adj = np.load(ALPHAS_NPY)  # Shape: (K, T)

# 2) SSH‐Tunnel und DB‐Verbindung
with sshtunnel(settings["ssh-tunnel"]["name"]):
    db = engine()
    register_schemas(db)

    n_profiles = len(id_array)
    n_batches = int(np.ceil(n_profiles / BATCH_SIZE))
    logger.info(f"Starte Kalibrierung in {n_batches} Batches à {BATCH_SIZE} Profiles.")

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, n_profiles)
        batch_ids = id_array[start:end]
        id_list_str = ",".join(map(str, batch_ids.tolist()))

        # 3) Lade Originalprofile batch‐weise
        query = f"""
            SELECT {ID_COLUMN}, {TS_COLUMN}
            FROM {TABLE_NAME}
            WHERE {ID_COLUMN} IN ({id_list_str})
            ORDER BY {ID_COLUMN}
        """
        df = pd.read_sql(query, db)

        # 4) Kalibriere jedes Profil
        def calibrate_row(row):
            cid = int(row[ID_COLUMN])
            raw = row[TS_COLUMN]
            # Falls raw bereits eine Liste ist, nicht json.loads aufrufen
            if isinstance(raw, str):
                ts = np.array(json.loads(raw), dtype=float)
            else:
                ts = np.array(raw, dtype=float)
            cl = id_to_cluster[cid]
            calibrated = ts * alphas_adj[cl]
            return (
                calibrated.tolist()
            )  # oder json.dumps(calibrated.tolist()) wenn Du wieder JSON willst

        df["load_in_wh_calibrated"] = df.apply(calibrate_row, axis=1)

        # 5) Speichern als CSV-Chunk
        out_path = os.path.join(OUTPUT_DIR, f"calibrated_batch_{b:03d}.csv")
        df[[ID_COLUMN, "load_in_wh_calibrated"]].to_csv(out_path, index=False)
        logger.info(f"Batch {b + 1}/{n_batches} gespeichert: {out_path}")

logger.info("Alle Profile kalibriert und gespeichert.")
