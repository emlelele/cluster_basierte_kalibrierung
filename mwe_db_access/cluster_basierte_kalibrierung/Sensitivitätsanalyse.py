#!/usr/bin/env python3
"""
Kalibrierung mit Box-Constraints (B1..B4) – konsistente Skalierung:
- SLP wird auf 3 922.20 kWh/Jahr pro Haushalt skaliert
- Kalibrierung erfolgt auf aggregierter Skala (Wh, multipliziert mit N)
- Globale Nachjustierung erzwingt N * 3 922.20 kWh/Jahr
- Kennzahlen & Plots werden pro Haushalt in kWh/h ausgewiesen
- Plots nur für B2..B4 (mit Nachjustierung), deutsch beschriftet
- NMBE zusätzlich explizit für UNCAL und B1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------------------------
# Pfade (ggf. anpassen)
# ----------------------------
MEAN_PROFILES_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_mean_full_ts.npy"
LABELS_CSV        = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_labels_full_ts.csv"
VALIDATION_CSV    = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"

OUT_DIR_PLOTS     = "./plots_bounds_adjust_only_kwh_phh"
OUT_DIR_ALPHAS    = "./alphas_bounds_adjust_only"
OUT_CSV_METRICS   = "./metrics_bounds_adjust_only_kwh_phh.csv"

os.makedirs(OUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUT_DIR_ALPHAS, exist_ok=True)

# ----------------------------
# A) Daten laden
# ----------------------------
mean_profiles = np.load(MEAN_PROFILES_NPY)   # Wh/h, Shape (K, T)
K, T = mean_profiles.shape
labels = pd.read_csv(LABELS_CSV)["cluster"].to_numpy()
counts = np.bincount(labels, minlength=K).astype(float)
N = float(counts.sum())  # Anzahl Profile/Haushalte

df_val = pd.read_csv(VALIDATION_CSV)
slp_kwh = df_val["Energy_kWh"].to_numpy(dtype=float)        # kWh/h
slp_wh  = slp_kwh * 1000.0                                  # Wh/h

# Aggregiertes Originalprofil (Wh/h)
agg_wh = (mean_profiles * counts[:, None]).sum(axis=0)      # (T,)

# ----------------------------
# B) Ziel: 3 922.20 kWh/Jahr pro Haushalt
# ----------------------------
TARGET_KWH_PER_HH = 3922.20
TARGET_WH_PER_HH  = TARGET_KWH_PER_HH * 1000.0
TARGET_WH_TOTAL   = TARGET_WH_PER_HH * N  # aggregiert (Wh/Jahr)

# SLP pro Haushalt auf Zielenergie skalieren (Wh/h)
scale_slp = TARGET_WH_PER_HH / slp_wh.sum()
slp_wh_phh = slp_wh * scale_slp          # Wh/h pro Haushalt (Zielsumme erreicht)

# Für Kalibrierung: SLP auf aggregierte Skala heben (Wh/h * N)
slp_wh_total = slp_wh_phh * N

# ----------------------------
# C) Kalibrierung pro Bounds + globale Nachjustierung auf TARGET_WH_TOTAL
# ----------------------------
def calibrate_bounds(bounds):
    lb, ub = bounds
    alphas = np.zeros((K, T), dtype=float)
    for t in range(T):
        A = (counts * mean_profiles[:, t]).reshape(1, K)  # aggregiert (Wh/h)
        b = np.array([slp_wh_total[t]])                   # aggregiert (Wh/h)
        res = lsq_linear(A, b,
                         bounds=(lb * np.ones(K), ub * np.ones(K)),
                         lsmr_tol="auto")
        alphas[:, t] = res.x

    # aggregierte Kalibrierungsergebnisse
    V_cal_bnd = (counts[:, None] * mean_profiles * alphas).sum(axis=0)  # Wh/h
    # globale Nachjustierung auf gewünschte Jahresenergie (aggregiert)
    gf = TARGET_WH_TOTAL / V_cal_bnd.sum() if V_cal_bnd.sum() != 0 else 1.0
    alphas_adj = alphas * gf
    V_cal_adj  = (counts[:, None] * mean_profiles * alphas_adj).sum(axis=0)  # Wh/h

    return alphas, alphas_adj, V_cal_bnd, V_cal_adj, gf

# ----------------------------
# D) Hilfsfunktionen (Metriken/Plots pro Haushalt in kWh/h)
# ----------------------------
HOURS_PER_WEEK = 168

def avg_week(series):
    n_full = (series.shape[0] // HOURS_PER_WEEK) * HOURS_PER_WEEK
    return series[:n_full].reshape(-1, HOURS_PER_WEEK).mean(axis=0)

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def pearson_corr(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def nmbe(y_true, y_pred):
    """ASHRAE: (Sum(y - yhat)) / ((n - p) * ybar) [%], p≈1"""
    n = len(y_true)
    ybar = np.mean(y_true)
    p = 1
    return float((np.sum(y_true - y_pred) / ((n - p) * ybar)) * 100.0)

def cvrmse(y_true, y_pred):
    """ASHRAE: CV(RMSE) in % mit (n - p), p≈1"""
    n = len(y_true)
    ybar = np.mean(y_true)
    p = 1
    rmse = np.sqrt(np.sum((y_pred - y_true)**2) / (n - p))
    return float((rmse / ybar) * 100.0)

def total_energy_kwh(series_wh_per_hh):
    return float(series_wh_per_hh.sum() / 1000.0)

def peak_info(series_wh_per_hh):
    peak_val_wh = float(series_wh_per_hh.max())
    peak_pos    = int(np.argmax(series_wh_per_hh)) + 1
    return peak_val_wh, peak_pos

def compute_metrics(tag, y_true_wh_phh, y_pred_wh_phh):
    """Kennzahlen pro Haushalt (Wh/h) – gibt u. a. CV(RMSE), NMBE in % zurück."""
    # Jahr (Wh/h)
    rmse_year  = float(np.sqrt(mean_squared_error(y_true_wh_phh, y_pred_wh_phh)))
    mae_year   = float(mean_absolute_error(y_true_wh_phh, y_pred_wh_phh))
    mape_year  = mape(y_true_wh_phh, y_pred_wh_phh)
    r_year     = pearson_corr(y_true_wh_phh, y_pred_wh_phh)
    nmbe_year  = nmbe(y_true_wh_phh, y_pred_wh_phh)
    cvrmse_year = cvrmse(y_true_wh_phh, y_pred_wh_phh)
    slp_E_kWh  = total_energy_kwh(y_true_wh_phh)
    pred_E_kWh = total_energy_kwh(y_pred_wh_phh)
    peak_pred_wh, peak_pred_h = peak_info(y_pred_wh_phh)
    peak_slp_wh , peak_slp_h  = peak_info(y_true_wh_phh)

    # Woche (Wh/h)
    week_true = avg_week(y_true_wh_phh)
    week_pred = avg_week(y_pred_wh_phh)
    rmse_week = float(np.sqrt(mean_squared_error(week_true, week_pred)))
    mae_week  = float(mean_absolute_error(week_true, week_pred))
    mape_week = mape(week_true, week_pred)
    r_week    = pearson_corr(week_true, week_pred)

    return {
        "Case": tag,
        "RMSE_year_Wh": rmse_year, "MAE_year_Wh": mae_year,
        "MAPE_year_%": mape_year, "CVRMSE_year_%": cvrmse_year,
        "NMBE_year_%": nmbe_year, "Pearson_year": r_year,
        "SLP_Total_kWh": slp_E_kWh, "PRED_Total_kWh": pred_E_kWh,
        "Total_Diff_kWh": pred_E_kWh - slp_E_kWh,
        "Total_Diff_%": ((pred_E_kWh - slp_E_kWh) / slp_E_kWh) * 100.0,
        "Peak_year_Wh_pred": peak_pred_wh, "Peak_year_hour_pred": peak_pred_h,
        "Peak_year_Wh_slp": peak_slp_wh,   "Peak_year_hour_slp": peak_slp_h,
        "RMSE_week_Wh": rmse_week, "MAE_week_Wh": mae_week,
        "MAPE_week_%": mape_week, "Pearson_week": r_week
    }

def plot_week(tag, uncal_wh_phh, pred_wh_phh, slp_wh_phh, out_png):
    """Plot mittlere Woche in kWh/h (pro Haushalt)."""
    # Umrechnen in kWh/h für die Achse
    u_kwh = avg_week(uncal_wh_phh) / 1000.0
    p_kwh = avg_week(pred_wh_phh)  / 1000.0
    s_kwh = avg_week(slp_wh_phh)   / 1000.0

    plt.figure(figsize=(12, 4))
    x = np.arange(1, HOURS_PER_WEEK + 1)
    plt.plot(x, u_kwh, label="Unkalibriert", linestyle=":", linewidth=1.2)
    plt.plot(x, p_kwh, label=f"Kalibriert (mit Nachjustierung) – {tag}", linewidth=1.6)
    plt.plot(x, s_kwh, label="SLP (skaliert)", linestyle="--", linewidth=1.2)
    plt.title(f"Mittleres Wochenprofil (168 h) – {tag}")
    plt.xticks(range(0, 169, 24), [str(i) for i in range(0, 169, 24)])
    plt.xlabel("Stunde der Woche")
    plt.ylabel("Last [kWh]")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

# ----------------------------
# E) Baselines (pro Haushalt)
# ----------------------------
uncal_wh_phh = agg_wh / N          # Wh/h pro Haushalt (unkalibriert, normiert)
slp_wh_phh   = slp_wh_phh          # Wh/h pro Haushalt (Zielsumme)

print(f"[INFO] N={int(N)} Profile, Zielenergie pro HH: {TARGET_KWH_PER_HH:.2f} kWh/Jahr")
print(f"[INFO] SLP check: {total_energy_kwh(slp_wh_phh):.2f} kWh/Jahr pro HH (soll 3922.20)")

# ----------------------------
# F) Bounds definieren
# ----------------------------
bounds_sets = {
    "B1 [0.8,1.2]": (0.8, 1.2),
    "B2 [0.7,1.3]": (0.7, 1.3),
    "B3 [0.6,1.4]": (0.6, 1.4),
    "B4 [0.5,1.5]": (0.5, 1.5),
}

# ----------------------------
# G) Kalibrieren, Metriken, Plots (nur B2..B4)
# ----------------------------
metrics = []

# UNCAL – Metriken pro Haushalt
metrics.append(compute_metrics("UNCAL", slp_wh_phh, uncal_wh_phh))

for tag, bounds in bounds_sets.items():
    alphas_bnd, alphas_adj, V_cal_bnd, V_cal_adj, gf = calibrate_bounds(bounds)

    # Speichern der Alphas
    base = tag.replace(" ", "").replace("[","").replace("]","").replace(",","_").replace(".","p")
    np.save(os.path.join(OUT_DIR_ALPHAS, f"alphas_{base}_bounded.npy"),  alphas_bnd)
    np.save(os.path.join(OUT_DIR_ALPHAS, f"alphas_{base}_adjusted.npy"), alphas_adj)

    # Aggregiert -> pro Haushalt
    pred_wh_phh = V_cal_adj / N

    # Metriken (pro Haushalt)
    metrics.append(compute_metrics(tag + " +Adjust", slp_wh_phh, pred_wh_phh))

    # NMBE zusätzlich ausgeben für B1
    if tag.startswith("B1"):
        nmbe_b1 = nmbe(slp_wh_phh, pred_wh_phh)
        print(f"[INFO] NMBE (Jahr, pro HH) für {tag} + Adjust: {nmbe_b1:.2f} %")

    # Plots NUR für B2..B4
    if tag.startswith("B2") or tag.startswith("B3") or tag.startswith("B4"):
        out_png = os.path.join(OUT_DIR_PLOTS, f"week_{base}_adjust_kwh_phh.png")
        plot_week(tag, uncal_wh_phh, pred_wh_phh, slp_wh_phh, out_png)
        print(f"[OK] Plot gespeichert: {out_png}")

# NMBE zusätzlich explizit für UNCAL (pro Haushalt)
from math import isfinite
nmbe_uncal = nmbe(slp_wh_phh, uncal_wh_phh)
print(f"[INFO] NMBE (Jahr, pro HH) für UNCAL: {nmbe_uncal:.2f} %")

# ----------------------------
# H) Ergebnisse speichern
# ----------------------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT_CSV_METRICS, index=False)
print(f"[OK] Kennzahlen gespeichert: {OUT_CSV_METRICS}")

# Konsolen-Auszug
cols = ["Case","RMSE_year_Wh","MAE_year_Wh","MAPE_year_%","CVRMSE_year_%","NMBE_year_%",
        "Pearson_year","SLP_Total_kWh","PRED_Total_kWh","Total_Diff_kWh","Total_Diff_%",
        "RMSE_week_Wh","MAE_week_Wh","MAPE_week_%","Pearson_week"]
print("\n=== Zusammenfassung (Auszug, pro Haushalt) ===")
print(metrics_df[cols].to_string(index=False, justify="left"))
