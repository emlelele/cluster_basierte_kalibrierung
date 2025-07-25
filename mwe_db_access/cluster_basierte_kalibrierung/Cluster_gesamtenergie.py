"""
Berechnet zunächst die Gesamtjahresenergie aller unkalibrierten Profile:
  1. Multipliziert jedes Cluster-Mittelprofil c_i(t) mit der Anzahl Profile |C_i|
  2. Summiert stündlich und über alle Cluster zur Gesamtenergie E_total (in kWh)

Bestimmt dann die Energie der Validierungsreihe V_val(t):
  3. Lädt die CSV unter '/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv'
  4. Summiert stündlich alle Werte zur Energie E_val (in kWh)

Ermittelt den Skalierungsfaktor α = E_total / E_val

Erzeugt die auf das Niveau von E_total hochskalierte Validierungs-Kurve:
  V_target(t) = α * V_val(t)
"""

import numpy as np
import pandas as pd

# 1) Pfade zu den Datendateien
LABELS_CSV = "cluster_labels_full_ts.csv"  # Spalten: id, cluster
MEAN_PROFILES_NPY = "cluster_mean_full_ts.npy"  # Array mit Form (K, T)
VALIDATION_CSV = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/slp_h25_2023_hourly.csv"


def main():
    # ----------------------------------------------------------------
    # Teil A: Gesamtenergie der unkalibrierten Cluster-Profile ermitteln
    # ----------------------------------------------------------------
    # 1.1) Labels und Mittelprofile laden
    df_labels = pd.read_csv(LABELS_CSV)
    labels = df_labels["cluster"].to_numpy()  # Länge N = 100 000
    mean_profiles = np.load(MEAN_PROFILES_NPY)  # Form: (K, T), in Wh

    # 1.2) Cluster-Größen zählen
    K, T = mean_profiles.shape
    counts = np.bincount(labels, minlength=K)  # Anzahl Profile pro Cluster

    # 1.3) Summen pro Cluster und Gesamtenergie berechnen
    #      cluster_sums[c,h] = mean_profiles[c,h] * counts[c]
    cluster_sums = mean_profiles * counts[:, None]  # Wh pro Cluster und Stunde
    energy_per_cluster = cluster_sums.sum(axis=1)  # Wh pro Cluster über alle Stunden
    total_energy_Wh = energy_per_cluster.sum()  # Wh aller Profile
    total_energy_kWh = total_energy_Wh / 1000.0  # in kWh

    print(f"Unkalibrierte Gesamtenergie aller Profile: {total_energy_kWh:,.0f} kWh")

    # ----------------------------------------------------------------
    # Teil B: Energie der Validierungsreihe berechnen
    # ----------------------------------------------------------------
    # 2.1) Validierungs-Kurve laden
    val_df = pd.read_csv(VALIDATION_CSV)
    hourly_vals = val_df.iloc[:, 1].to_numpy()  # Annahme: 2. Spalte = kWh-Werte

    # 2.2) Jahresenergie der Validierung ermitteln
    val_total_kWh = hourly_vals.sum()
    print(f"Originale Validierungsenergie: {val_total_kWh:,.0f} kWh")

    # ----------------------------------------------------------------
    # Teil C: Skalierungsfaktor und Zielkurve
    # ----------------------------------------------------------------
    # 3.1) Faktor bestimmen, damit Validierung auf E_total skaliert wird
    scale_factor = (total_energy_kWh / val_total_kWh) / 100000
    print(f"Skalierungsfaktor: {scale_factor:.6f}")

    # 3.2) Zielkurve erstellen (auf E_total-Niveau)
    V_target = hourly_vals * scale_factor
    print(f"Validierungsenergie nach Skalierung: {V_target.sum():,.0f} kWh")


if __name__ == "__main__":
    main()
