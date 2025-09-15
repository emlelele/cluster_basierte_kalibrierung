Clusterbasierte Kalibrierung synthetischer Haushaltslasten

Dieses Repo kalibriert synthetische Haushaltslastprofile (≈100 000 Profile aus einer DB) gegen ein Referenz-SLP (BDEW H25).
Ablauf: Profile clustern → pro Stunde per Least-Squares (mit/ohne Bounds) skalieren → globale Nachjustierung der Jahresenergie (optional) → Faktoren auf alle Einzelprofile anwenden → Sensitivität & Validierung (Plots + Metriken).

Inhalte & Skripte

Clustering

cluster_basierte_kalibrierung/Cluster.py
MiniBatch-KMeans auf voller Jahreszeitreihe (T=8760).
Output: cluster_labels_full_ts.csv, cluster_mean_full_ts.npy

cluster_basierte_kalibrierung_PAA/cluster_paa.py
Clustering auf PAA-Features (Dim.-Reduktion via tslearn).
Output: cluster_labels_paa.csv, cluster_centers_paa.npy, cluster_mean_full_paa.npy

Kalibrierung (α-Matrizen, K×T)

least_squares_kalibrierung.py
Zeitvariabler LSQ auf aggregierter Skala, default-Bounds 0.8…1.2.
Output: calibration_matrix.npy, alphas_full_matrix.csv

kalibrierung_ohne_bounds.py
LSQ ohne Bounds.
Output: calibration_matrix_unbounded.npy, alphas_unbounded.csv

kalibrierung_mit_nachjustierung.py
Bounds 0.8…1.2 plus globale Nachjustierung der Jahresenergie.
Output: calibration_matrix_bounded_adjusted.npy, alphas_bounded_adjusted.csv

Anwendung & Validierung

einzelprofil_kalibrierung.py
Wendet α(t) je Cluster batchweise auf alle Einzelprofile an.
Output: calibrated_profiles_batches/calibrated_batch_###.csv

Sensitivitätsanalyse.py
Bounds-Szenarien (z. B. [0.8,1.2]…[0.5,1.5]), globale Nachjustierung, Kennzahlen & Wochenplots.
Output: alphas_bounds_adjust_only/*.npy, plots_bounds_adjust_only_kwh_phh/*.png, metrics_bounds_adjust_only_kwh_phh.csv

Validierung.py
Stichproben-Validierung: mittlere Woche (SLP vs. unkalibriert vs. kalibriert), Jahres/Wochen-Metriken, Peaks.
Output: validation_plots/validation_mean_week_rep_#.png, validation_plots/validation_metrics_*.csv

Pipeline (End-to-End)

Clustering

# Vollprofil
python mwe_db_access/cluster_basierte_kalibrierung/Cluster.py
# oder: PAA
python mwe_db_access/cluster_basierte_kalibrierung_PAA/cluster_paa.py


Kalibrierung (α bestimmen)

# Empfohlen: Bounds + Nachjustierung
python mwe_db_access/cluster_basierte_kalibrierung/kalibrierung_mit_nachjustierung.py
# Alternativen:
# python mwe_db_access/cluster_basierte_kalibrierung/least_squares_kalibrierung.py
# python mwe_db_access/cluster_basierte_kalibrierung/kalibrierung_ohne_bounds.py


α anwenden (alle Einzelprofile)

python mwe_db_access/cluster_basierte_kalibrierung/einzelprofil_kalibrierung.py


Sensitivität (optional)

python mwe_db_access/cluster_basierte_kalibrierung/Sensitivitätsanalyse.py


Validierung (Stichprobe, Plots, Kennzahlen)

python mwe_db_access/cluster_basierte_kalibrierung/Validierung.py

Voraussetzungen

Python ≥ 3.10

Pakete: numpy, pandas, scipy, scikit-learn, tslearn, matplotlib, loguru

pip install numpy pandas scipy scikit-learn tslearn matplotlib loguru


Interne Module: mwe_db_access.config, mwe_db_access.db, mwe_db_access.ssh
→ In settings SSH-Tunnel-Name & DB-Zugang setzen; register_schemas() muss die benötigten Schemas laden.

Datenquellen

DB-Tabelle demand.iee_household_load_profiles mit Spalten
id (int), load_in_wh (Array/JSON mit 8760 Wh-Werten).

SLP: mwe_db_access/data/slp_h25_2023_hourly.csv mit Spalte Energy_kWh (8760 Zeilen).

Hinweis: Einige Skripte enthalten absolute Pfade (z. B. /home/emre/...). Bitte am Skriptkopf anpassen oder zentral über eine Config/ENV-Variablen lösen.

Wichtige Standard-Parameter

Clustering: K=10, BATCH_SIZE=500, N_EPOCHS=2, RANDOM_STATE=42

Kalibrierung (bounded): Box-Bounds typ. 0.8…1.2 (Sensitivität bis 0.5…1.5)

Anwendung auf Profile: BATCH_SIZE=1000 (Export der kalibrierten Profile)

Outputs (Erwartung)

Cluster: cluster_labels_*.csv (id→cluster), cluster_mean_full_*.npy (K×8760)

Kalibrierung: calibration_matrix*.npy (K×8760), alphas_*.csv (8760×K)

Kalibrierte Profile: calibrated_profiles_batches/calibrated_batch_###.csv (id, load_in_wh_calibrated)

Plots & Metriken: Wochenplots (PNG), Kennzahlen (CSV) inkl. RMSE/MAE/MAPE, Pearson, CV(RMSE), NMBE, Peaks, Energiesummen

Tipps & Troubleshooting

Performance/RAM: Batch-Größen (BATCH_SIZE) passend wählen; MiniBatchKMeans ist streaming-fähig.

SSH/DB: Skripte öffnen selbst einen Tunnel (settings["ssh-tunnel"]["name"]); gültige Keys/Agent sicherstellen.

Energie-Konsistenz: Skripte skalieren SLP auf Zielenergie; Variante mit Nachjustierung erzwingt exakte Jahressumme nach LSQ.

Pfad-Hygiene: Absolute Pfade vermeiden (gemeinsames BASE_DIR/ENV hilft).
