"""
Dieses Skript lädt vorgeclusterte Lastprofile, berechnet für jede Stunde eines Jahres
Kalibrierungsfaktoren α_i(t) für jedes Cluster mittels eines zeitvariablen
Least-Squares-Abgleichs und gibt am Ende eine CSV-Datei aus, in der jede Zeile
einer Stunde (8760 Zeilen) und jede Spalte einem der 10 Cluster entspricht.

A) Laden der Mittelprofile und Berechnung der Cluster-Gewichte
B) Bestimmung der unkalibrierten Jahresenergie pro Haushalt
C) Einlesen und Normierung der Validierungskurve (Variante A)
D) Zeitvariabler Least-Squares-Abgleich zur Bestimmung von α (Form: K×T)
E) Speichern der Kalibrierungsmatrix als .npy und Export als CSV (8760×10)
"""

import numpy as np
import pandas as pd

from scipy.optimize import lsq_linear

# ----------------------------
# Pfade zu Deinen Datendateien
# ----------------------------
MEAN_PROFILES_NPY = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/cluster_mean_full_ts.npy"
LABELS_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/cluster_labels_full_ts.csv"
VALIDATION_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"


def main():
    # ----------------------------------
    # A) Cluster-Daten laden & Gewichte
    # ----------------------------------
    # 1) Mittelprofile c_i(t) laden (Wh); Form: (K, T)
    mean_profiles = np.load(MEAN_PROFILES_NPY)
    K, T = mean_profiles.shape  # K = Anzahl Cluster, T = Stunden pro Jahr (8760)

    # 2) Cluster-Labels einlesen und Häufigkeiten ermitteln
    labels = pd.read_csv(LABELS_CSV)["cluster"].to_numpy()
    counts = np.bincount(labels, minlength=K)  # |C_i| für jedes Cluster i
    N = labels.size  # Gesamtzahl Haushalte

    # 3) Relative Gewichte w_i = |C_i| / N
    weights = counts / N

    # ---------------------------------------------
    # B) Unkalibrierte Jahresenergie pro Haushalt
    # ---------------------------------------------
    # 1) Aggregierter Verbrauch aller Haushalte je Stunde (Wh)
    agg_wh = (mean_profiles * counts[:, None]).sum(axis=0)
    # 2) Durchschnittsprofil je Haushalt (Wh/Stunde)
    V_val_profile_wh = agg_wh / N
    # 3) Jahresenergie pro Haushalt in kWh
    E_total_kWh = V_val_profile_wh.sum() / 1000.0

    print("=== Abschnitt B: Unkalibrierte Jahresenergie ===")
    print(f"E_total_kWh (pro Haushalt): {E_total_kWh:.2f} kWh\n")

    # # ----------------------------------
    # # C) Validierungs-Kurve laden & normieren (Variante A)
    # # ----------------------------------
    # val_df    = pd.read_csv(VALIDATION_CSV)
    # # ACHTUNG: val_df["Energy_kWh"] ist in kWh/Stunde über alle Haushalte
    # V_raw_agg = val_df["Energy_kWh"].to_numpy()       # kWh/Stunde gesamt
    # V_raw     = V_raw_agg / N                         # kWh/Stunde je Haushalt
    # # Jahresenergie pro Haushalt
    # E_val_kWh = V_raw.sum()
    #
    # # Skalenfaktor, damit beide Jahresenergien übereinstimmen
    # scale    = E_total_kWh / E_val_kWh
    # V_target = V_raw * scale                          # kWh/Stunde je Haushalt
    #
    # print("=== Abschnitt C: Validierung & Debug ===")
    # print(f"E_val_kWh   (pro Haushalt): {E_val_kWh:.2f} kWh")
    # print(f"Skalenfaktor: {scale:.3f}\n")
    #
    # print("Stunde | V_val_profile_wh (Wh/h) | V_raw_per_house (Wh/h) | V_target (Wh/h)")
    # for t in range(min(5, T)):
    #     print(f"{t:6d} | "
    #           f"{V_val_profile_wh[t]:22.2f} | "
    #           f"{(V_raw[t]*1000):22.2f} | "
    #           f"{(V_target[t]*1000):22.2f}")
    # print()
    #
    # # -------------------------------------------------
    # # D) Zeitvariabler Least-Squares: Matrix α (K, T)
    # # -------------------------------------------------
    # # Initialisiere Kalibrierungsmatrix
    # alphas = np.zeros((K, T), dtype=np.float64)
    #
    # for t in range(T):
    #     # A_row[i] = w_i * c_i(t) in Wh/Stunde
    #     A_row = weights * mean_profiles[:, t]
    #     A     = A_row.reshape(1, K)            # Form (1 x K)
    #     b     = np.array([V_target[t] * 1000]) # Ziel in Wh/Stunde
    #
    #     # Box-Bounds für α_i(t)
    #     lb = 0.8 * np.ones(K)
    #     ub = 1.2 * np.ones(K)
    #
    #     # LSQ mit Box-Beschränkung
    #     res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
    #     alphas[:, t] = res.x
    #
    #     if t % 1000 == 0:
    #         print(f" → Stunde {t+1}/{T} kalibriert")

    # ---------------------
    # C) Aggregierte Normierung
    # ---------------------
    val_df = pd.read_csv(VALIDATION_CSV)
    # 1) Roh-SLP in kWh/Stunde (aggregiert über alle Haushalte)
    slp_kwh = val_df["Energy_kWh"].to_numpy()  # Länge 8760
    # 2) In Wh/Stunde umrechnen
    slp_wh = slp_kwh * 1000.0  # Wh/Stunde
    # 3) Aggregierter Cluster-Verbrauch ebenfalls in Wh/Stunde
    agg_wh = (mean_profiles * counts[:, None]).sum(axis=0)  # Wh/Stunde
    # 4) Skalenfaktor so, dass Summe(agg_wh) = Summe(slp_wh_scaled)
    scale = agg_wh.sum() / slp_wh.sum()
    slp_scaled_wh = slp_wh * scale  # Wh/Stunde target

    # -------------------------
    # D) Zeitvariabler LSQ mit Aggregaten
    # -------------------------
    alphas = np.zeros((K, T), dtype=float)
    for t in range(T):
        # A_row[i] = counts[i] * c_i(t) in Wh/Stunde
        A_row = counts * mean_profiles[:, t]
        A = A_row.reshape(1, K)  # (1×K)
        b = np.array([slp_scaled_wh[t]])  # Wh/Stunde target

        lb = 0.8 * np.ones(K)
        ub = 1.2 * np.ones(K)
        res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
        alphas[:, t] = res.x

        if t % 1000 == 0:
            print(f" → Stunde {t + 1}/{T} kalibriert")

    # ----------------------------------------
    # E) Ergebnisse speichern & als CSV ausgeben
    # ----------------------------------------
    # 1) Speichere die rohe Kalibrierungsmatrix
    np.save("calibration_matrix.npy", alphas)
    print("\nKalibrierungsmatrix α (K×T) gespeichert in 'calibration_matrix.npy'")

    # 2) Exportiere als CSV mit Form (8760 Zeilen × 10 Spalten)
    #    Zeilen: Stunden h0…h8759, Spalten: cluster_0…cluster_{K-1}
    df_alphas = pd.DataFrame(
        alphas.T,
        index=[f"h{t}" for t in range(T)],
        columns=[f"cluster_{i}" for i in range(K)],
    )
    out_csv = "alphas_full_matrix.csv"
    df_alphas.to_csv(out_csv, index=True)
    print(f"CSV-Datei erzeugt: {out_csv} (Shape: {df_alphas.shape})")


if __name__ == "__main__":
    main()
