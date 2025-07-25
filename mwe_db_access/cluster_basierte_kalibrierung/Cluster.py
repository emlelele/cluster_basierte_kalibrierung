import os

import numpy as np
import pandas as pd

from loguru import logger
from sklearn.cluster import MiniBatchKMeans

from mwe_db_access.config import settings
from mwe_db_access.db import engine, register_schemas
from mwe_db_access.ssh import sshtunnel

# --------------------------------------------
# PARAMETER (anpassen nach Bedarf)
# --------------------------------------------
K = 10  # Anzahl der Cluster (z.B. 20–30)
BATCH_SIZE = 500  # Anzahl Profile pro Batch (RAM-Limite beachten)
N_EPOCHS = 2  # Wie oft man über alle Batches „partial_fit“ macht
RANDOM_STATE = 42  # Zufalls-Seed für Reproduzierbarkeit
VERBOSE = True  # Ausgabe von Fortschrittsmeldungen an/aus


# --------------------------------------------
# HAUPTFUNKTION: Clustern aller vollen Jahresprofile
# --------------------------------------------
def cluster_full_time_series(
    k: int = K,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
    random_state: int = RANDOM_STATE,
):
    """
    1) Liest alle Profil-IDs (n ≈ 100 000).
    2) Bestimmt T (Jahreslänge eines Profils), indem man ein Beispielprofil lädt.
    3) Initialisiert MiniBatchKMeans mit n_clusters=k und parametern.
    4) Führt „n_epochs“ über den Datensatz durch, indem man in Batches
       jeweils die vollen Zeitreihen (Länge T) lädt und mit partial_fit(train_batch) trainiert.
    5) Nach dem Training haben wir Centroids (k × T).
    6) In einem zweiten Durchlauf: alle Profile wieder batch-weise laden,
       Labels = kmeans.predict(matrix) berechnen und parallel:
       • Summen-Array cluster_sum_full[k, T] aufaddieren
       • cluster_count[k] inkrementieren
    7) Aus (Summe / Count) pro Cluster = echte Mittelwert-Jahresprofil (k × T).
    Return:
        cluster_labels: numpy-Array (n_profiles,) mit Label 0…k−1 (in ID-Reihenfolge)
        cluster_centers_full: numpy-Array (k, T) mit den KMeans-Centroids (volle Länge)
        cluster_mean_full: numpy-Array (k, T) mit dem tatsächlichen Durchschnittsprofil je Cluster
    """
    # 1) SSH-Tunnel & DB-Verbindung aufbauen
    with sshtunnel(settings["ssh-tunnel"]["name"]):
        if VERBOSE:
            logger.info("SSH-Tunnel aufgebaut, DB-Verbindung...")
        db_engine = engine()
        register_schemas(db_engine)

        # 2) Alle Profil-IDs aus der DB holen
        if VERBOSE:
            logger.info("Lade alle Profil-IDs aus DB…")
        all_ids_df = pd.read_sql(
            "SELECT id FROM demand.iee_household_load_profiles", db_engine
        )
        all_ids = all_ids_df["id"].to_numpy()  # Array der Länge n_profiles
        n_profiles = len(all_ids)
        if VERBOSE:
            logger.info(f"Anzahl Profile insgesamt: {n_profiles}")

        # 3) Länge T der Zeitreihen ermitteln (z.B. 8760), indem wir einmal ein Profil abfragen
        probe_id = int(all_ids[0])
        probe_q = f"""
            SELECT load_in_wh
            FROM demand.iee_household_load_profiles
            WHERE id = {probe_id}
        """
        probe_df = pd.read_sql(probe_q, db_engine)
        ts0 = np.asarray(probe_df["load_in_wh"].iloc[0], dtype=np.float32)
        T = ts0.shape[0]
        if VERBOSE:
            logger.info(f"Jahreszeitreihe-Länge pro Profil = {T} Stunden.")

        # 4) MiniBatchKMeans initialisieren (Clustering auf volle Länge-T)
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,  # Batch-Größe steuert internen Update-Step
            max_no_improvement=10,
            verbose=0,
        )

        # 5) Mehrere Epochen: Batches durchlaufen und partial_fit auf volle Arrays anwenden
        n_batches = int(np.ceil(n_profiles / batch_size))
        if VERBOSE:
            logger.info(
                f"Starte MiniBatch-KMeans Training über {n_epochs} Epochen, {n_batches} Batches/Epoche."
            )
        for epoch in range(n_epochs):
            if VERBOSE:
                logger.info(f" Epoch {epoch+1}/{n_epochs}")
            # Wir können die IDs optional randomisieren, um jeden Epoch-Durchlauf anders zu mischen:
            rng = np.random.default_rng(random_state + epoch)
            permuted_ids = rng.permutation(all_ids)
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_profiles)
                batch_ids = permuted_ids[start_idx:end_idx]
                id_list_str = ",".join(map(str, batch_ids.tolist()))

                query = f"""
                    SELECT id, load_in_wh
                    FROM demand.iee_household_load_profiles
                    WHERE id IN ({id_list_str})
                    ORDER BY id
                """
                df_batch = pd.read_sql(query, db_engine)

                # Matrix der Form (current_batch_size, T) erzeugen
                # Achtung: ORDER BY id liefert die gleiche Reihenfolge wie batch_ids.sort(),
                # aber wir haben permuted_ids auf Positionen, daher ordnen wir explizit jedem
                # DataFrame-Row seine Position in der Batch-Matrix zu:
                id_to_pos = {pid: idx for idx, pid in enumerate(batch_ids)}
                cur_size = len(batch_ids)
                mtx = np.zeros((cur_size, T), dtype=np.float32)
                for _, row in df_batch.iterrows():
                    pid = int(row["id"])
                    ts = np.asarray(row["load_in_wh"], dtype=np.float32)
                    pos = id_to_pos[pid]
                    # Sollte ts.shape[0] == T sein, sonst Fehler
                    if ts.shape[0] != T:
                        raise RuntimeError(
                            f"Profil {pid} hat Länge {ts.shape[0]} ≠ erwartete {T}."
                        )
                    mtx[pos, :] = ts

                # MiniBatchKMeans-Update schrittweise
                mbk.partial_fit(mtx)

                if VERBOSE and (i + 1) % 10 == 0:
                    logger.info(
                        f"   Epoch {epoch+1}, Batch {i+1}/{n_batches} verarbeitet."
                    )

        # 6) Nach Abschluss: die Centroid-Profile (k, T)
        cluster_centers_full = mbk.cluster_centers_.astype(np.float64)  # shape (k, T)
        if VERBOSE:
            logger.info(
                "MiniBatchKMeans Training abgeschlossen. Centroids liegen als volle Länge T vor."
            )

        # 7) Jetzt Labels für alle 100 000 Profile berechnen (batch-weise Predict)
        cluster_labels = np.empty(n_profiles, dtype=np.int32)
        if VERBOSE:
            logger.info("Berechne Cluster-Label für jedes Profil (batch-weise kallk.)…")
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_profiles)
            batch_ids = all_ids[start_idx:end_idx]
            id_list_str = ",".join(map(str, batch_ids.tolist()))

            query = f"""
                SELECT id, load_in_wh
                FROM demand.iee_household_load_profiles
                WHERE id IN ({id_list_str})
                ORDER BY id
            """
            df_batch = pd.read_sql(query, db_engine)

            id_to_pos = {pid: idx for idx, pid in enumerate(batch_ids)}
            cur_size = len(batch_ids)
            mtx = np.zeros((cur_size, T), dtype=np.float32)
            for _, row in df_batch.iterrows():
                pid = int(row["id"])
                ts = np.asarray(row["load_in_wh"], dtype=np.float32)
                pos = id_to_pos[pid]
                mtx[pos, :] = ts

            labels_batch = mbk.predict(mtx)  # (cur_size,)
            for pid, pos in id_to_pos.items():
                global_pos = start_idx + pos
                cluster_labels[global_pos] = labels_batch[pos]

            if VERBOSE and (i + 1) % 10 == 0:
                logger.info(
                    f"   Label-Bestimmung Batch {i+1}/{n_batches} abgeschlossen."
                )

        # 8) Nun for each Cluster das echte Durchschnittsprofil (T) berechnen:
        #    Summenarray (k, T) und Zähler (k,)
        cluster_sum_full = np.zeros((k, T), dtype=np.float64)
        cluster_count = np.zeros(k, dtype=np.int64)

        if VERBOSE:
            logger.info(
                "Summiere in einem zweiten Durchlauf alle Profile pro Cluster auf…"
            )
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_profiles)
            batch_ids = all_ids[start_idx:end_idx]
            id_list_str = ",".join(map(str, batch_ids.tolist()))

            query = f"""
                SELECT id, load_in_wh
                FROM demand.iee_household_load_profiles
                WHERE id IN ({id_list_str})
                ORDER BY id
            """
            df_batch = pd.read_sql(query, db_engine)

            id_to_pos = {pid: idx for idx, pid in enumerate(batch_ids)}
            cur_size = len(batch_ids)
            mtx = np.zeros((cur_size, T), dtype=np.float64)
            for _, row in df_batch.iterrows():
                pid = int(row["id"])
                ts = np.asarray(row["load_in_wh"], dtype=np.float64)
                pos = id_to_pos[pid]
                mtx[pos, :] = ts

            # Labels für dieses Batch (haben wir zuvor bereits komplett berechnet)
            for idx_in_batch, pid in enumerate(batch_ids):
                global_pos = start_idx + idx_in_batch
                lbl = int(cluster_labels[global_pos])
                # Summiere
                cluster_sum_full[lbl] += mtx[idx_in_batch, :]
                cluster_count[lbl] += 1

            if VERBOSE and (i + 1) % 10 == 0:
                logger.info(f"   Summen-Batch {i+1}/{n_batches} abgeschlossen.")

        # 9) Echte Mittelwerte pro Cluster = Summe / Count
        cluster_mean_full = np.zeros_like(cluster_sum_full)
        for c in range(k):
            if cluster_count[c] > 0:
                cluster_mean_full[c, :] = cluster_sum_full[c, :] / cluster_count[c]
            else:
                if VERBOSE:
                    logger.warning(
                        f"Cluster {c} ist leer – setze Mittelwertprofile auf 0."
                    )
                cluster_mean_full[c, :] = np.zeros(T, dtype=np.float64)

        if VERBOSE:
            logger.info("Alle Mittelwert-Jahresprofile pro Cluster berechnet.")
        # 10) Speicherung DER RESULTATE INNERHALB DES TUNNELS
        # -----------------------------------------
        # a) Cluster-Labels + IDs in CSV
        df_out = pd.DataFrame(
            {
                "id": pd.read_sql(
                    "SELECT id FROM demand.iee_household_load_profiles ORDER BY id",
                    db_engine,
                )["id"].to_numpy(),
                "cluster": cluster_labels,
            }
        )
        df_out.to_csv("cluster_labels_full_ts.csv", index=False)
        logger.info("Cluster-Labels in 'cluster_labels_full_ts.csv' gespeichert.")

        # b) Mittelwert-Profile als .npy
        np.save("cluster_mean_full_ts.npy", cluster_mean_full)
        logger.info("Mittelwert-Jahresprofile (cluster_mean_full_ts.npy) gespeichert.")
    # Ende SSH-Tunnel
    return cluster_labels, cluster_centers_full, cluster_mean_full


if __name__ == "__main__":
    cluster_full_time_series(
        k=K, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, random_state=RANDOM_STATE
    )
# # --------------------------------------------
# # BEISPIEL: Skriptmässige Ausführung
# # --------------------------------------------
# if __name__ == "__main__":
#     labels, centroids, mean_profiles = cluster_full_time_series(
#         k           = K,
#         batch_size  = BATCH_SIZE,
#         n_epochs    = N_EPOCHS,
#         random_state= RANDOM_STATE,
#     )
#
#     # Optional: Speichere Cluster-Labels mit den dazugehörigen IDs
#     # Achte darauf, dass ORDER BY id in df_out dieselbe Reihenfolge hat wie all_ids
#     df_out = pd.DataFrame({
#         "id":       pd.read_sql(
#                         "SELECT id FROM demand.iee_household_load_profiles ORDER BY id",
#                         engine()
#                     )["id"].to_numpy(),
#         "cluster":  labels
#     })
#     df_out.to_csv("cluster_labels_full_ts.csv", index=False)
#     logger.info("Cluster-Labels in 'cluster_labels_full_ts.csv' gespeichert.")
#
#     # Optional: Speichere echte Mittelwerte (k, T) lokal (z.B. .npy)
#     np.save("cluster_mean_full_ts.npy", mean_profiles)
#     logger.info("Mittelwert-Jahresprofile (cluster_mean_full_ts.npy) gespeichert.")
