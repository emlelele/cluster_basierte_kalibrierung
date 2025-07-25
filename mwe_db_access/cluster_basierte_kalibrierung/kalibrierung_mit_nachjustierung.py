"""
Kalibrierung mit Box-Constraints und globaler Nachjustage

Dieses Skript:
 1. Lädt Mittelprofile und Clustergrößen.
 2. Skaliert SLP auf aggregierte Cluster-Jahresenergie.
 3. Führt stundenweise LSQ unter Beschränkung 0.8≤α_i≤1.2 durch.
 4. Berechnet anschließend einen globalen Faktor, um verbleibende
    Jahresenergie-Differenz auszugleichen.
 5. Speichert die angepasste α-Matrix (.npy) und als CSV.
"""
import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear

# Pfade
MEAN_PROFILES_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_mean_full_ts.npy"
LABELS_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_labels_full_ts.csv"
VALIDATION_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"

# A) Laden
mean_profiles = np.load(MEAN_PROFILES_NPY)  # (K, T) Wh/Stunde
K, T = mean_profiles.shape
labels = pd.read_csv(LABELS_CSV)["cluster"].to_numpy()
counts = np.bincount(labels, minlength=K)

# B) SLP aggregiert normieren
df_val = pd.read_csv(VALIDATION_CSV)
slp_wh = df_val["Energy_kWh"].to_numpy() * 1000.0
agg_wh = (mean_profiles * counts[:, None]).sum(axis=0)
scale = agg_wh.sum() / slp_wh.sum()
slp_scaled_wh = slp_wh * scale

# C) LSQ mit Box-Bounds 0.8–1.2
alphas = np.zeros((K, T), dtype=float)
for t in range(T):
    A_row = counts * mean_profiles[:, t]
    A = A_row.reshape(1, K)
    b = np.array([slp_scaled_wh[t]])
    # Bounds-Limits definieren
    lb = 0.8 * np.ones(K)
    ub = 1.2 * np.ones(K)
    # LSQ mit Bounds: min ||A·α - b||^2  s.t. 0.8≤α≤1.2
    res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
    alphas[:, t] = res.x
    if (t + 1) % 1000 == 0:
        print(f" → Stunde {t+1}/{T} kalibriert")

# D) Globale Nachjustage
# 1) Energie vor Kalibrierung (kWh/Jahr)
E_uncal = agg_wh.sum() / 1000.0
# 2) Energie nach LSQ (kWh/Jahr)
V_cal = (counts[:, None] * mean_profiles * alphas).sum(axis=0)
E_cal = V_cal.sum() / 1000.0
# 3) Faktor zum Angleichen
global_factor = E_uncal / E_cal
# Auf α anwenden
alphas_adjusted = alphas * global_factor
print(f"E_orig={E_uncal:.3f}kWh, E_post={E_cal:.3f}kWh, factor={global_factor:.6f}")

# E) Speichern der finalen α-Matrix
np.save("calibration_matrix_bounded_adjusted.npy", alphas_adjusted)
df_adj = pd.DataFrame(
    alphas_adjusted.T,
    index=[f"h{t}" for t in range(T)],
    columns=[f"cluster_{i}" for i in range(K)],
)
df_adj.to_csv("alphas_bounded_adjusted.csv", index=True)
print("[Bounded+Adjusted] α-Matrix gespeichert in .npy und .csv.")
