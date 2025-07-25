"""
Kalibrierung ohne Box-Constraints (unbounded)

Dieses Skript:
 1. Lädt vorgeclusterte Lastprofile (Wh/Stunde) und Clustergrößen
 2. Liest die Validierungskurve (SLP) ein und skaliert sie auf die gleiche
    Jahresenergie wie das aggregierte Cluster-Profil.
 3. Führt pro Stunde eine unbeschränkte Least-Squares-Optimierung durch, um
    α_i(t) zu finden, so dass die Summe aller Cluster-Beiträge punktgenau
    dem skalierten SLP-Wert entspricht.
 4. Speichert die α-Matrix (.npy) und exportiert sie als CSV (8760×K).
"""
import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear

# ----------------------------
# A) Daten laden
# ----------------------------
# Mittelprofile: c_i(t) in Wh/Stunde, Shape (K, T)
MEAN_PROFILES_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_mean_full_ts.npy"
# Labels: Cluster-Zugehörigkeit jedes Haushalts, wird zur Häufigkeit gezählt
LABELS_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_labels_full_ts.csv"
# Validierung: Standardlastprofil SLP in kWh/Stunde
VALIDATION_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"

# 1) Lade Mittelprofile c_i(t) in Wh/Stunde
mean_profiles = np.load(MEAN_PROFILES_NPY)
K, T = mean_profiles.shape

# 2) Bestimme Cluster-Größen |C_i|
labels = pd.read_csv(LABELS_CSV)["cluster"].to_numpy()
counts = np.bincount(labels, minlength=K)

# ----------------------------
# B) SLP aggregiert normieren
# ----------------------------
val_df = pd.read_csv(VALIDATION_CSV)
slp_kwh = val_df["Energy_kWh"].to_numpy()  # kWh/Stunde gesamt
slp_wh = slp_kwh * 1000.0  # Wh/Stunde gesamt

# Aggregierter Cluster-Verbrauch (Wh/Stunde)
agg_wh = (mean_profiles * counts[:, None]).sum(axis=0)

# Skalenfaktor, damit SUM(agg_wh) == SUM(slp_wh_scaled)
scale = agg_wh.sum() / slp_wh.sum()
slp_scaled_wh = slp_wh * scale  # Zielprofil in Wh/Stunde

# ----------------------------
# C) Unbeschränkter LSQ-Loop
# ----------------------------
# Definiere unbeschränkte Bounds per Komponente
lb = -np.inf * np.ones(K)
ub = +np.inf * np.ones(K)

# Initialisiere Kalibrierungsmatrix
alphas = np.zeros((K, T), dtype=float)

for t in range(T):
    # A_row: Beitrag jedes Clusters (Wh/Stunde)
    A_row = counts * mean_profiles[:, t]
    A = A_row.reshape(1, K)  # Form (1×K)
    b = np.array([slp_scaled_wh[t]])  # Zielwert (Wh/Stunde)

    # LSQ ohne Grenzen: min ||A·α - b||^2
    res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
    alphas[:, t] = res.x

    if (t + 1) % 1000 == 0:
        print(f" → Stunde {t+1}/{T} kalibriert")

# ----------------------------
# D) Ergebnisse speichern
# ----------------------------
np.save("calibration_matrix_unbounded.npy", alphas)
df = pd.DataFrame(
    alphas.T,
    index=[f"h{t}" for t in range(T)],
    columns=[f"cluster_{i}" for i in range(K)],
)
df.to_csv("alphas_unbounded.csv", index=True)
print("[Unbounded] α-Matrix gespeichert in .npy und .csv.")
