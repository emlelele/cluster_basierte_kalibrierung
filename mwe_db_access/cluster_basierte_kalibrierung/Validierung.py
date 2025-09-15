

"""
VALIDIERUNG (nur Wochenplots):
- Drei Kurven: SLP (skaliert auf 3922.199 kWh), UNKALIBRIERT (auch auf 3922.199 kWh skaliert), KALIBRIERT (Stichproben-Mittel)
- 5 Wiederholungen, je CSV 5 Zufallsprofile
- Plot: Mittlere Woche (y-Limit 0..1200 Wh)
- Kennzahlen: RMSE, MAE, MAPE, Pearson, Peaks, Energie-Diff (für Jahr & Woche)
"""

import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PARAMETER / PFADE
# =========================
CSV_DIR = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/calibrated_profiles_batches"
SLP_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"  # Spalte: Energy_kWh
TARGET_TOTAL_KWH = 3922.199  # Zieljahresenergie für SLP UND UNCAL

MEAN_PROFILES_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_mean_full_ts.npy"
LABELS_CSV        = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/cluster_basierte_kalibrierung/cluster_labels_full_ts.csv"

SAVE_DIR = "./validation_plots"

SAMPLES_PER_FILE = 5
N_REPEATS = 5
RANDOM_SEED = 42

HOURS_PER_WEEK = 168
T_EXPECTED = 8760
Y_LIMIT = (0, 1200)        # feste y-Achse
FIGSIZE_WEEK = (12, 3.8)

AM_WINDOW = (6, 16)     # 06–16 (0-basiert)
PM_WINDOW = (16, 22)    # 16–21 (0-basiert)


# =========================
# HILFSFUNKTIONEN
# =========================
def load_slp_scaled(slp_csv_path: str, target_total_kwh: float) -> np.ndarray:
    """SLP (Energy_kWh) laden, in Wh/h umrechnen und auf target_total_kwh/Jahr skalieren."""
    df_val = pd.read_csv(slp_csv_path)
    if "Energy_kWh" not in df_val.columns:
        raise ValueError("Erwarte Spalte 'Energy_kWh' im SLP-CSV.")
    slp_wh = df_val["Energy_kWh"].to_numpy(dtype=float) * 1000.0
    factor = (target_total_kwh * 1000.0) / slp_wh.sum()
    slp_wh = slp_wh * factor
    print(f"[INFO] SLP skaliert auf {target_total_kwh:.3f} kWh/Jahr (Faktor={factor:.6f})")
    return slp_wh


def build_uncalibrated_baseline_scaled(mean_profiles_npy: str, labels_csv: str, target_total_kwh: float) -> np.ndarray:
    """
    Unkalibrierte Referenz erzeugen und auf target_total_kwh/Jahr skalieren:
      uncal(t) = Sum_i |C_i| * c_i(t)
      anschließend Energieskalierung auf Zieljahresenergie (wie beim SLP).
    """
    mean_profiles = np.load(mean_profiles_npy)  # (K, T)
    K, T = mean_profiles.shape
    labels = pd.read_csv(labels_csv)["cluster"].to_numpy()
    counts = np.bincount(labels, minlength=K).astype(float)  # (K,)
    uncal = (mean_profiles * counts[:, None]).sum(axis=0)    # (T,)

    if uncal.shape[0] != T_EXPECTED:
        raise ValueError(f"Uncal length {uncal.shape[0]} != {T_EXPECTED}")

    # auf gleiche Jahresenergie wie SLP skalieren
    target_wh = target_total_kwh * 1000.0
    factor = target_wh / uncal.sum()
    uncal = uncal * factor
    print(f"[INFO] UNCAL skaliert auf {target_total_kwh:.3f} kWh/Jahr (Faktor={factor:.6f})")
    return uncal


def mean_week(profile: np.ndarray) -> np.ndarray:
    n_full_weeks = profile.shape[0] // HOURS_PER_WEEK
    used = profile[: n_full_weeks * HOURS_PER_WEEK].reshape(n_full_weeks, HOURS_PER_WEEK)
    return used.mean(axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))) * 100.0)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def peak_info_week(profile: np.ndarray, label: str) -> Dict[str, float]:
    am_start, am_end = AM_WINDOW
    pm_start, pm_end = PM_WINDOW
    am_slice = profile[am_start:am_end]
    pm_slice = profile[pm_start:pm_end]
    am_val = float(am_slice.max()); am_idx = am_start + int(np.argmax(am_slice))
    pm_val = float(pm_slice.max()); pm_idx = pm_start + int(np.argmax(pm_slice))
    return {
        f"{label}_AM_Peak_Wh": am_val,
        f"{label}_AM_Peak_hour": am_idx + 1,
        f"{label}_PM_Peak_Wh": pm_val,
        f"{label}_PM_Peak_hour": pm_idx + 1
    }


def peak_info_year(profile: np.ndarray, label: str) -> Dict[str, float]:
    peak_val = float(profile.max())
    peak_pos = int(np.argmax(profile)) + 1
    return {f"{label}_Year_Peak_Wh": peak_val, f"{label}_Year_Peak_hour": peak_pos}


def total_energy_kwh(profile_wh: np.ndarray) -> float:
    return float(profile_wh.sum() / 1000.0)


def load_and_sample_profiles(csv_files: List[str], samples_per_file: int, rng: np.random.Generator, T: int) -> np.ndarray:
    sampled = []
    for path in csv_files:
        df = pd.read_csv(path)
        if "load_in_wh_calibrated" not in df.columns:
            raise ValueError(f"Spalte 'load_in_wh_calibrated' fehlt in {path}")
        if len(df) < samples_per_file:
            raise ValueError(f"Zu wenige Zeilen in {path} für {samples_per_file} Stichproben.")
        idxs = rng.choice(len(df), size=samples_per_file, replace=False)
        rows = df.iloc[idxs]["load_in_wh_calibrated"]
        for js in rows:
            arr = np.array(json.loads(js), dtype=float) if isinstance(js, str) else np.array(js, dtype=float)
            if arr.shape[0] != T:
                raise ValueError(f"Länge des Profils != {T} in {path}")
            sampled.append(arr)
    return np.vstack(sampled)


def plot_week(slp_w: np.ndarray, mean_cal_w: np.ndarray, uncal_w: np.ndarray, out_path: str, title_suffix: str):
    """Mittlere Woche, drei Kurven, y-Limit 0..1200 Wh, 24h-Ticks."""
    hours = np.arange(1, HOURS_PER_WEEK + 1)
    plt.figure(figsize=FIGSIZE_WEEK)
    plt.plot(hours, slp_w,      label="SLP – mittlere Woche",          linewidth=1.4)
    plt.plot(hours, uncal_w,    label="unkalibriert – mittlere Woche", linewidth=1.2)
    plt.plot(hours, mean_cal_w, label="kalibriert – mittlere Woche",   linewidth=1.8)
    plt.title(f"Mittlere Woche: SLP, unkalibriert, kalibriert – {title_suffix}")
    plt.xlabel("Stunde der Woche (1–168)")
    plt.ylabel("Last [Wh]")
    plt.ylim(*Y_LIMIT)
    ax = plt.gca()
    ax.set_xticks([1] + list(range(24, HOURS_PER_WEEK + 1, 24)))
    ax.set_xticks(range(6, HOURS_PER_WEEK + 1, 6), minor=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# =========================
# HAUPTPROGRAMM
# =========================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) SLP & UNCAL auf gleiche Jahresenergie skalieren
    slp_wh   = load_slp_scaled(SLP_CSV, TARGET_TOTAL_KWH)
    T = slp_wh.shape[0]
    assert T == T_EXPECTED, f"Erwarte {T_EXPECTED} Stunden, bekommen {T}."

    uncal_wh = build_uncalibrated_baseline_scaled(MEAN_PROFILES_NPY, LABELS_CSV, TARGET_TOTAL_KWH)

    # 2) Dateien einsammeln
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSVs in {CSV_DIR} gefunden.")
    print(f"[INFO] {len(csv_files)} CSV-Dateien gefunden.")

    # 3) Vorab: mittlere Woche SLP/UNCAL
    slp_week   = mean_week(slp_wh)
    uncal_week = mean_week(uncal_wh)

    metrics_rows = []

    # 4) Wiederholungen
    for rep in range(1, N_REPEATS + 1):
        print(f"\n[RUN {rep}/{N_REPEATS}] Stichproben werden gezogen …")
        rng = np.random.default_rng(RANDOM_SEED + rep)

        # 4.1) Kalibrierte Stichprobe -> Jahresmittel & Wochenmittel
        T = slp_wh.shape[0]
        sampled_profiles = load_and_sample_profiles(csv_files, SAMPLES_PER_FILE, rng, T)
        mean_cal_year = sampled_profiles.mean(axis=0)
        mean_cal_week = mean_week(mean_cal_year)

        # ===== Kennzahlen: KALIBRIERT vs SLP =====
        row = {"repeat": rep}

        # Jahr
        row["CAL_RMSE_year_Wh"] = rmse(slp_wh, mean_cal_year)
        row["CAL_MAE_year_Wh"]  = mae(slp_wh, mean_cal_year)
        row["CAL_MAPE_year_%"]  = mape(slp_wh, mean_cal_year)
        row["CAL_Pearson_year"] = pearson_corr(slp_wh, mean_cal_year)
        row["SLP_Total_kWh"]    = total_energy_kwh(slp_wh)
        row["CAL_Total_kWh"]    = total_energy_kwh(mean_cal_year)
        row["CAL_Total_Diff_kWh"] = row["CAL_Total_kWh"] - row["SLP_Total_kWh"]
        row["CAL_Total_Diff_%"]   = (row["CAL_Total_Diff_kWh"] / row["SLP_Total_kWh"]) * 100.0

        # Woche
        row["CAL_RMSE_week_Wh"] = rmse(slp_week, mean_cal_week)
        row["CAL_MAE_week_Wh"]  = mae(slp_week, mean_cal_week)
        row["CAL_MAPE_week_%"]  = mape(slp_week, mean_cal_week)
        row["CAL_Pearson_week"] = pearson_corr(slp_week, mean_cal_week)

        # Peaks
        row.update(peak_info_week(slp_week, "SLP"))
        row.update(peak_info_week(mean_cal_week, "CAL"))
        row.update(peak_info_year(slp_wh, "SLP"))
        row.update(peak_info_year(mean_cal_year, "CAL"))
        row["CAL_AM_Peak_Diff_Wh"] = row["CAL_AM_Peak_Wh"] - row["SLP_AM_Peak_Wh"]
        row["CAL_PM_Peak_Diff_Wh"] = row["CAL_PM_Peak_Wh"] - row["SLP_PM_Peak_Wh"]

        # ===== Kennzahlen: UNKALIBRIERT vs SLP (ebenfalls auf 3922.199 kWh skaliert) =====
        # Jahr
        row["UNCAL_RMSE_year_Wh"] = rmse(slp_wh, uncal_wh)
        row["UNCAL_MAE_year_Wh"]  = mae(slp_wh, uncal_wh)
        row["UNCAL_MAPE_year_%"]  = mape(slp_wh, uncal_wh)
        row["UNCAL_Pearson_year"] = pearson_corr(slp_wh, uncal_wh)
        row["UNCAL_Total_kWh"]    = total_energy_kwh(uncal_wh)
        row["UNCAL_Total_Diff_kWh"] = row["UNCAL_Total_kWh"] - row["SLP_Total_kWh"]  # ~0 by construction
        row["UNCAL_Total_Diff_%"]   = (row["UNCAL_Total_Diff_kWh"] / row["SLP_Total_kWh"]) * 100.0

        # Woche
        row["UNCAL_RMSE_week_Wh"] = rmse(slp_week, uncal_week)
        row["UNCAL_MAE_week_Wh"]  = mae(slp_week, uncal_week)
        row["UNCAL_MAPE_week_%"]  = mape(slp_week, uncal_week)
        row["UNCAL_Pearson_week"] = pearson_corr(slp_week, uncal_week)

        # Peaks (unkalibriert)
        row.update(peak_info_week(uncal_week, "UNCAL"))
        row.update(peak_info_year(uncal_wh, "UNCAL"))
        row["UNCAL_AM_Peak_Diff_Wh"] = row["UNCAL_AM_Peak_Wh"] - row["SLP_AM_Peak_Wh"]
        row["UNCAL_PM_Peak_Diff_Wh"] = row["UNCAL_PM_Peak_Wh"] - row["SLP_PM_Peak_Wh"]

        metrics_rows.append(row)

        # 4.2) Plot – mittlere Woche, drei Kurven
        plot_week(
            slp_w=slp_week,
            mean_cal_w=mean_cal_week,
            uncal_w=uncal_week,
            out_path=os.path.join(SAVE_DIR, f"validation_mean_week_rep_{rep}.png"),
            title_suffix=f"Stichprobe #{rep}"
        )

    # 5) Ergebnisse speichern
    metrics_df = pd.DataFrame(metrics_rows)
    per_rep_path = os.path.join(SAVE_DIR, "validation_metrics_repeats_with_uncal.csv")
    metrics_df.to_csv(per_rep_path, index=False)
    print(f"[OK] Metriken je Wiederholung gespeichert: {per_rep_path}")

    summary = metrics_df.drop(columns=["repeat"]).agg(["mean", "std"])
    summary_path = os.path.join(SAVE_DIR, "validation_metrics_summary_with_uncal.csv")
    summary.to_csv(summary_path)
    print(f"[OK] Zusammenfassung (Mittel/Std) gespeichert: {summary_path}")

if __name__ == "__main__":
    main()
