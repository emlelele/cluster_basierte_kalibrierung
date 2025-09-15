import os

import numpy as np
import pandas as pd

from loguru import logger
from sklearn.cluster import MiniBatchKMeans
from tslearn.piecewise import (
    PiecewiseAggregateApproximation,  # PAA-Transformer
)

from mwe_db_access.config import settings
from mwe_db_access.db import engine, register_schemas
from mwe_db_access.ssh import sshtunnel

# --------------------------------------------
# PARAMETER (anpassen nach Bedarf)
# --------------------------------------------
K = 10  # Anzahl der Cluster
N_SEGMENTS = 730  # Anzahl PAA-Blöcke (neue Dimensionalität)
BATCH_SIZE = 500  # Anzahl Profile pro Batch (RAM-Limit)
N_EPOCHS = 2  # Wie oft man über alle Batches partial_fit macht
RANDOM_STATE = 42  # Seed für Reproduzierbarkeit
VERBOSE = True  # Progress-Logging an/aus


# --------------------------------------------
# HAUPTFUNKTION: Clustern auf PAA-Features
# --------------------------------------------
def cluster_full_time_series_paa(
    k: int = K,
    n_segments: int = N_SEGMENTS,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
    random_state: int = RANDOM_STATE,
):
    """
    1) Liest alle Profil-IDs (n ≈ 100 000).
    2) Bestimmt T (8760) und initialisiert PAA(n_segments).
    3) Initialisiert MiniBatchKMeans auf PAA-Feature-Raum.
    4) Batch-weise:
         a) Lade Rohprofile (batch_size × T)
         b) Wende PAA an → (batch_size × n_segments)
         c) mbk.partial_fit(paa_batch)
    5) Nach Training: Centroids im PAA-Raum (k × n_segments).
    6) Batch-weise:
         a) Lade Rohprofile
         b) PAA → paa_batch
         c) labels = mbk.predict(paa_batch)
         d) kumuliere RAW-Profile für Mittelprofil-Berechnung
    7) Berechne echte Mittelprofile (k × T).
    Return:
        cluster_labels      : np.ndarray (n_profiles,)
        cluster_centers_paa : np.ndarray (k, n_segments)
        cluster_mean_full   : np.ndarray (k, T)
    """

    # 1) SSH-Tunnel & DB-Verbindung aufbauen
    with sshtunnel(settings["ssh-tunnel"]["name"]):
        if VERBOSE:
            logger.info("→ SSH-Tunnel & DB-Verbindung aufgebaut")
        db_engine = engine()
        register_schemas(db_engine)

        # 2a) Profil-IDs laden
        all_ids_df = pd.read_sql(
            "SELECT id FROM demand.iee_household_load_profiles ORDER BY id", db_engine
        )
        all_ids = all_ids_df["id"].to_numpy()
        n_profiles = len(all_ids)
        if VERBOSE:
            logger.info(f"→ {n_profiles} Profile gefunden")

        # 2b) Jahreslänge T ermitteln
        probe_id = int(all_ids[0])
        probe_df = pd.read_sql(
            f"""
            SELECT load_in_wh FROM demand.iee_household_load_profiles
            WHERE id = {probe_id}
        """,
            db_engine,
        )
        ts0 = np.asarray(probe_df["load_in_wh"].iloc[0], dtype=np.float32)
        T = ts0.shape[0]
        if VERBOSE:
            logger.info(f"→ Profil-Länge T = {T} Stunden")

        # 3) PAA initialisieren
        paa = PiecewiseAggregateApproximation(n_segments=n_segments)
        paa = PiecewiseAggregateApproximation(n_segments=n_segments)
        # damit paa.transform() später funktioniert, brauchen wir einmal paa.fit()
        # wir nutzen dazu unser Beispielprofil ts0 (Form: (T,))
        paa.fit(ts0[np.newaxis, :, np.newaxis])
        # 4) MiniBatchKMeans auf PAA-Features initialisieren
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            max_no_improvement=10,
            verbose=0,
        )

        n_batches = int(np.ceil(n_profiles / batch_size))

        # --- 4) TRAINING ---
        if VERBOSE:
            logger.info(f"→ Starte Training: {n_epochs} Epochen × {n_batches} Batches")
        for epoch in range(n_epochs):
            if VERBOSE:
                logger.info(f"  Epoch {epoch+1}/{n_epochs}")
            rng = np.random.default_rng(random_state + epoch)
            permuted_ids = rng.permutation(all_ids)
            for i in range(n_batches):
                # lade batch
                batch_ids = permuted_ids[i * batch_size : (i + 1) * batch_size]
                ids_str = ",".join(map(str, batch_ids))
                df = pd.read_sql(
                    f"""
                    SELECT id, load_in_wh FROM demand.iee_household_load_profiles
                    WHERE id IN ({ids_str}) ORDER BY id
                """,
                    db_engine,
                )

                # Roh-Matrix (batch_size × T)
                mtx_raw = np.zeros((len(batch_ids), T), dtype=np.float32)
                pos = {pid: idx for idx, pid in enumerate(batch_ids)}
                for _, row in df.iterrows():
                    pid = int(row["id"])
                    mtx_raw[pos[pid], :] = np.asarray(
                        row["load_in_wh"], dtype=np.float32
                    )

                # PAA: (batch_size, T, 1) → (batch_size, n_segments, 1) → squeeze
                mtx_paa = paa.transform(mtx_raw[:, :, np.newaxis]).squeeze(axis=2)

                # partial_fit im PAA-Raum
                mbk.partial_fit(mtx_paa)

                if VERBOSE and (i + 1) % 10 == 0:
                    logger.info(f"    Training: Batch {i+1}/{n_batches} fertig")

        # Cluster-Zentren im PAA-Raum
        cluster_centers_paa = mbk.cluster_centers_

        # --- 5) LABELING & MITTELPROFIL ---
        cluster_labels = np.empty(n_profiles, dtype=np.int32)
        cluster_sum = np.zeros((k, T), dtype=np.float64)
        cluster_count = np.zeros(k, dtype=np.int64)

        if VERBOSE:
            logger.info("→ Berechne Labels + echte Mittelprofile")
        for i in range(n_batches):
            # lade batch
            batch_ids = all_ids[i * batch_size : (i + 1) * batch_size]
            ids_str = ",".join(map(str, batch_ids))
            df = pd.read_sql(
                f"""
                SELECT id, load_in_wh FROM demand.iee_household_load_profiles
                WHERE id IN ({ids_str}) ORDER BY id
            """,
                db_engine,
            )

            # Roh-Matrix
            mtx_raw = np.zeros((len(batch_ids), T), dtype=np.float32)
            pos = {pid: idx for idx, pid in enumerate(batch_ids)}
            for _, row in df.iterrows():
                pid = int(row["id"])
                mtx_raw[pos[pid], :] = np.asarray(row["load_in_wh"], dtype=np.float32)

            # PAA für Predict
            mtx_paa = paa.transform(mtx_raw[:, :, np.newaxis]).squeeze(axis=2)

            # Vorhersage
            labels_batch = mbk.predict(mtx_paa)
            for j, lbl in enumerate(labels_batch):
                # speichere Label
                global_idx = i * batch_size + j
                cluster_labels[global_idx] = lbl
                # kumuliere für Mittelprofil
                cluster_sum[lbl] += mtx_raw[j, :]
                cluster_count[lbl] += 1

            if VERBOSE and (i + 1) % 10 == 0:
                logger.info(f"    Label-Batch {i+1}/{n_batches} fertig")

        # arithm. Mittelprofil (k × T)
        cluster_mean_full = np.zeros_like(cluster_sum)
        for c in range(k):
            if cluster_count[c] > 0:
                cluster_mean_full[c] = cluster_sum[c] / cluster_count[c]

        # --- 6) ERGEBNISSE SPEICHERN ---
        pd.DataFrame({"id": all_ids, "cluster": cluster_labels}).to_csv(
            "cluster_labels_paa.csv", index=False
        )
        np.save("cluster_centers_paa.npy", cluster_centers_paa)
        np.save("cluster_mean_full_paa.npy", cluster_mean_full)

        if VERBOSE:
            logger.info(
                "→ Ergebnisse gespeichert: "
                "cluster_labels_paa.csv, cluster_centers_paa.npy, cluster_mean_full_paa.npy"
            )

    return cluster_labels, cluster_centers_paa, cluster_mean_full


if __name__ == "__main__":
    cluster_full_time_series_paa(
        k=K,
        n_segments=N_SEGMENTS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        random_state=RANDOM_STATE,
    )
