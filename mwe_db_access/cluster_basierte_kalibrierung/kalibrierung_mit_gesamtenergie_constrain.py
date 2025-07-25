#!/usr/bin/env python3
"""
Kalibrierung mit Box-Constraints und integrierter Jahresenergie-Constraint

Dieses Skript:
 1. Lädt Cluster-Mittelprofile und -Gewichte.
 2. Skaliert die Validierungskurve (SLP) auf aggregierte Cluster-Jahresenergie.
 3. Löst ein einziges constrained Least-Squares-Problem über alle Stunden,
    inklusive Bound-Constraints (0.8 ≤ α_i(t) ≤ 1.2) und einer Gleichheitsbedingung,
    die sicherstellt, dass die Jahresenergie exakt erhalten bleibt.
    - Vektorisiertes Problem
    - Solver: ECOS (Interieur-Punkt) oder OSQP mit angepassten Parametern
 4. Prüft die Bound- und Gleichheitsverletzungen und druckt Min/Max von α.
 5. Speichert die gefundene α-Matrix als .npy und CSV.

Benötigt: cvxpy, numpy, pandas

Usage:
    python calibration_with_annual_constraint.py
"""
import cvxpy as cp
import numpy as np
import pandas as pd

# ----------------------------
# Pfade (anpassen)
# ----------------------------
MEAN_PROFILES_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_mean_full_ts.npy"
LABELS_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_labels_full_ts.csv"
VALIDATION_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"

# A) Daten laden
mean_profiles = np.load(MEAN_PROFILES_NPY)  # (K, T) in Wh
K, T = mean_profiles.shape
labels = pd.read_csv(LABELS_CSV)["cluster"].to_numpy()
counts = np.bincount(labels, minlength=K)  # |C_i|

# B) SLP laden und normieren
slp_kwh = pd.read_csv(VALIDATION_CSV)["Energy_kWh"].to_numpy()  # kWh/Stunde
slp_wh = slp_kwh * 1000.0  # Wh/Stunde
agg_wh = (mean_profiles * counts[:, None]).sum(axis=0)  # Wh/Stunde
global_sum = agg_wh.sum()
scale = global_sum / slp_wh.sum()
slp_scaled_wh = slp_wh * scale  # Wh/Stunde target

# C) CVXPY-Variablen und vectorisiertes Problem
alpha = cp.Variable((K, T))  # α_i(t)
A = counts[:, None] * mean_profiles  # (K, T)

# Residual-Vektor über alle Stunden: r[t] = sum_i A[i,t]*α[i,t] - slp_scaled_wh[t]
r = cp.sum(cp.multiply(A, alpha), axis=0) - slp_scaled_wh
# Zielfunktion
obj = cp.Minimize(cp.sum_squares(r))
# Jahresenergie-Constraint
annual_constraint = cp.sum(cp.multiply(A, alpha)) == global_sum
# Box-Constraints
bounds = [alpha >= 0.8, alpha <= 1.2]

constraints = [annual_constraint] + bounds

# D) Problem lösen: OSQP mit angepassten Parametern
prob = cp.Problem(obj, constraints)
# ECOS nicht verfügbar, daher OSQP mit strengeren Toleranzen
prob.solve(
    solver=cp.OSQP,
    eps_abs=1e-6,
    eps_rel=1e-6,
    eps_prim_inf=1e-6,
    eps_dual_inf=1e-6,
    max_iter=50000,
    verbose=True,
)

# E) Validierung der Lösung
alphas_opt = alpha.value
print("Bounds-Check → min α:", alphas_opt.min(), "max α:", alphas_opt.max())
# Kontrolle Gleichheitsverletzung
diff = (A * alphas_opt).sum() - global_sum
print(f"Jahresenergie-Differenz (Wh): {diff:.6f}")
print("Optimization status:", prob.status)

# F) Speichern
np.save("calibration_matrix_annual_constraint.npy", alphas_opt)
df = pd.DataFrame(
    alphas_opt.T,
    index=[f"h{t}" for t in range(T)],
    columns=[f"cluster_{i}" for i in range(K)],
)
df.to_csv("alphas_annual_constraint.csv", index=True)
print("Gespeichert: .npy und .csv")
