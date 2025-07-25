import os

import numpy as np
import pandas as pd

from loguru import logger
from tslearn.clustering import TimeSeriesKMeans

# tslearn für PAA, DTW, K-Means mit DBA
from tslearn.piecewise import PiecewiseAggregateApproximation

from mwe_db_access.config import settings
from mwe_db_access.db import engine, register_schemas
from mwe_db_access.ssh import sshtunnel

# --------------------------------------------
# PARAMETER (anpassen nach Bedarf)
# --------------------------------------------
K = 10  # Anzahl der Cluster
N_SEGMENTS = 50  # Anzahl PAA-Blöcke (Dimension nach PAA)
BATCH_SIZE = 500  # Batch-Größe für DB-Lesezugriffe
RANDOM_STATE = 42  # Zufalls-Seed für Reproduzierbarkeit
VERBOSE = True  # Progress-Logging an/aus


def _reconstruct_from_paa(paa_series: np.ndarray, T: int) -> np.ndarray:
    """
    Rekonstruiert ein Vollprofil der Länge T aus einem PAA-Vektor.
    Wenn T % n_segments != 0, werden die ersten 'remainder' Segmente um 1 verlängert.
    """
    n_segments = paa_series.shape[0]
    base = T // n_segments
    remainder = T % n_segments
    lengths = [(base + 1 if i < remainder else base) for i in range(n_segments)]
    return np.concatenate(
        [np.repeat(paa_series[i], lengths[i]) for i in range(n_segments)]
    )


def cluster_with_paa_dtw_kmeans_dba(
    k: int = K,
    n_segments: int = N_SEGMENTS,
    batch_size: int = BATCH_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Pipeline: PAA -> DTW-KMeans mit DBA -> Berechnung echter Mittelprofile

    Return:
        cluster_labels      : np.ndarray (n_profiles,) mit Label 0…k−1
        cluster_centers_full: np.ndarray (k, T) mit den k Zentroiden als Vollprofile
        cluster_mean_full   : np.ndarray (k, T) mit arithmetischem Mittel je Cluster
    """
    # 1) SSH-Tunnel öffnen & DB-Verbindung
    with sshtunnel(settings["ssh-tunnel"]["name"]):
        if VERBOSE:
            logger.info("SSH-Tunnel & DB-Verbindung aufgebaut")
        db_engine = engine()
        register_schemas(db_engine)

        # IDs und Profil-Länge ermitteln
        all_ids_df = pd.read_sql(
            "SELECT id FROM demand.iee_household_load_profiles ORDER BY id", db_engine
        )
        all_ids = all_ids_df["id"].to_numpy()
        n_profiles = len(all_ids)
        if VERBOSE:
            logger.info(f"Anzahl Profile: {n_profiles}")

        # Probe-Profil für T
        probe_id = int(all_ids[0])
        probe_df = pd.read_sql(
            f"SELECT load_in_wh FROM demand.iee_household_load_profiles WHERE id = {probe_id}",
            db_engine,
        )
        ts0 = np.asarray(probe_df["load_in_wh"].iloc[0], dtype=np.float32)
        T = ts0.shape[0]
        if VERBOSE:
            logger.info(f"Profil-Länge T = {T} Stunden")

        # 2) PAA initialisieren
        paa = PiecewiseAggregateApproximation(n_segments=n_segments)
        paa_profiles = np.zeros((n_profiles, n_segments), dtype=np.float32)

        # 3) Batch-weise PAA-Transformation
        n_batches = int(np.ceil(n_profiles / batch_size))
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_profiles)
            batch_ids = all_ids[start:end]
            ids_str = ",".join(map(str, batch_ids.tolist()))

            df = pd.read_sql(
                f"SELECT id, load_in_wh FROM demand.iee_household_load_profiles WHERE id IN ({ids_str}) ORDER BY id",
                db_engine,
            )
            mtx = np.zeros((len(batch_ids), T), dtype=np.float32)
            idx_map = {pid: idx for idx, pid in enumerate(batch_ids)}
            for _, row in df.iterrows():
                pid = int(row["id"])
                mtx[idx_map[pid], :] = np.asarray(row["load_in_wh"], dtype=np.float32)

            # PAA: (n_ts, T, 1) -> (n_ts, n_segments, 1)
            mtx3d = mtx[:, :, np.newaxis]
            paa_batch = paa.fit_transform(mtx3d).squeeze(axis=2)
            paa_profiles[start:end, :] = paa_batch

            if VERBOSE and (i + 1) % 10 == 0:
                logger.info(f"PAA Batch {i+1}/{n_batches} fertig")

        # 4) DTW-KMeans mit DBA
        paa_profiles_3d = paa_profiles[:, :, np.newaxis]
        kmeans = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            init="k-means++",
            max_iter=30,
            random_state=random_state,
            verbose=VERBOSE,
        )
        cluster_labels = kmeans.fit_predict(paa_profiles_3d)
        compressed_centers = kmeans.cluster_centers_.squeeze(axis=2)
        if VERBOSE:
            logger.info("DTW-KMeans (DBA) abgeschlossen")

        # 5) Zentroiden als Vollprofile rekonstruieren
        centers_full = np.vstack(
            [_reconstruct_from_paa(compressed_centers[c], T) for c in range(k)]
        )

        # 6) Echte Mittel-Profile je Cluster berechnen
        cluster_sum = np.zeros((k, T), dtype=np.float64)
        cluster_count = np.zeros(k, dtype=np.int64)
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_profiles)
            batch_ids = all_ids[start:end]
            ids_str = ",".join(map(str, batch_ids.tolist()))

            df = pd.read_sql(
                f"SELECT id, load_in_wh FROM demand.iee_household_load_profiles WHERE id IN ({ids_str}) ORDER BY id",
                db_engine,
            )
            mtx = np.zeros((len(batch_ids), T), dtype=np.float32)
            idx_map = {pid: idx for idx, pid in enumerate(batch_ids)}
            for _, row in df.iterrows():
                pid = int(row["id"])
                mtx[idx_map[pid], :] = np.asarray(row["load_in_wh"], dtype=np.float32)

            labels_batch = cluster_labels[start:end]
            for j, lbl in enumerate(labels_batch):
                cluster_sum[lbl] += mtx[j]
                cluster_count[lbl] += 1

            if VERBOSE and (i + 1) % 10 == 0:
                logger.info(f"Mean-Summe Batch {i+1}/{n_batches}")

        cluster_mean_full = np.zeros_like(cluster_sum)
        for c in range(k):
            if cluster_count[c] > 0:
                cluster_mean_full[c] = cluster_sum[c] / cluster_count[c]
            else:
                logger.warning(f"Cluster {c} leer – setze Mittelprofil auf 0")

        if VERBOSE:
            logger.info("Alle echten Mittel-Profile berechnet")

        # Ergebnisse speichern
        pd.DataFrame({"id": all_ids, "cluster": cluster_labels}).to_csv(
            "cluster_labels_dtw_kmeans.csv", index=False
        )
        np.save("cluster_centers_dtw_kmeans.npy", centers_full)
        np.save("cluster_mean_dtw_kmeans.npy", cluster_mean_full)
        if VERBOSE:
            logger.info(
                "Ergebnisse gespeichert: cluster_labels_dtw_kmeans.csv, cluster_centers_dtw_kmeans.npy, cluster_mean_dtw_kmeans.npy"
            )

    return cluster_labels, centers_full, cluster_mean_full


if __name__ == "__main__":
    cluster_with_paa_dtw_kmeans_dba(
        k=K, n_segments=N_SEGMENTS, batch_size=BATCH_SIZE, random_state=RANDOM_STATE
    )
