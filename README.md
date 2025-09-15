# Clusterbasierte Kalibrierung synthetischer Haushaltslasten

Kalibriere \~100 000 synthetische Haushaltslastprofile gegen ein Referenz-SLP (BDEW H25):
**Clustern → stundenweise Least-Squares (mit/ohne Bounds) → optionale Jahresenergie-Nachjustierung → Anwendung auf alle Profile → Sensitivität & Validierung.**

---

## Inhaltsverzeichnis

* [Überblick](#überblick)
* [Pipeline (End-to-End)](#pipeline-end-to-end)
* [Schnellstart](#schnellstart)
* [Projektstruktur](#projektstruktur)
* [Skripte im Detail](#skripte-im-detail)
* [Voraussetzungen](#voraussetzungen)
* [Konfiguration & Datenquellen](#konfiguration--datenquellen)
* [Standard-Parameter](#standard-parameter)
* [Outputs](#outputs)
* [Tipps & Troubleshooting](#tipps--troubleshooting)

---

## Überblick

**Pipeline in 6 Schritten**

1. DB → Profile laden (`iee_household_load_profiles`)
2. Clustering (MiniBatchKMeans) → K Cluster
3. Cluster-Mittelprofile (Kx8760)
4. Kalibrierung alpha(t) via Least-Squares (mit/ohne Bounds)
5. Globale Jahresenergie-Nachjustierung (optional)
6. Anwendung auf alle Profile → CSV-Batches; anschließend Sensitivität & Validierung

```text
DB (iee_household_load_profiles)
  -> Clustering (MiniBatchKMeans)
  -> Cluster means (Kx8760)
  -> Calibration alpha(t) via least squares
  -> Global energy adjustment (optional)
  -> Apply to all profiles (CSV batches)
  -> Sensitivity & validation (plots, metrics)

* **Ziel:** Formtreue Kalibrierung großer synthetischer Datensätze gegenüber einem Referenz-SLP.
* **Kernidee:** Cluster-Mittelprofile dienen als Basis; pro Stunde werden skalierende Faktoren `α_i(t)` für jedes Cluster bestimmt und auf alle Einzelprofile zurückgespielt.

---

## Pipeline (End-to-End)

1. **Clustern** (Vollprofil *oder* PAA-Features)
2. **Kalibrieren** (`α` bestimmen; mit/ohne Bounds; optional globale Nachjustierung)
3. **α anwenden** (auf alle Einzelprofile, Batch-Export)
4. **Sensitivität** (Bounds-Szenarien) – *optional*
5. **Validierung** (Stichprobe, Wochenplots, Kennzahlen)

---

## Schnellstart

```bash
# 1) Clustering (Vollprofil)
python mwe_db_access/cluster_basierte_kalibrierung/Cluster.py
#   Alternative: PAA-Clustering
# python mwe_db_access/cluster_basierte_kalibrierung_PAA/cluster_paa.py

# 2) Kalibrierung (empfohlen: Bounds + Nachjustierung)
python mwe_db_access/cluster_basierte_kalibrierung/kalibrierung_mit_nachjustierung.py
#   Alternativen:
# python mwe_db_access/cluster_basierte_kalibrierung/least_squares_kalibrierung.py
# python mwe_db_access/cluster_basierte_kalibrierung/kalibrierung_ohne_bounds.py

# 3) α anwenden auf alle Einzelprofile (Batch-Export)
python mwe_db_access/cluster_basierte_kalibrierung/einzelprofil_kalibrierung.py

# 4) Sensitivität (optional)
python mwe_db_access/cluster_basierte_kalibrierung/Sensitivitätsanalyse.py

# 5) Validierung (Stichprobe, Plots, Kennzahlen)
python mwe_db_access/cluster_basierte_kalibrierung/Validierung.py
```

---

## Projektstruktur

```
mwe_db_access/
├─ cluster_basierte_kalibrierung/
│  ├─ Cluster.py                          # Clustering auf Vollprofil (T=8760)
│  ├─ least_squares_kalibrierung.py       # LSQ (Bounds), Aggregat-Skala
│  ├─ kalibrierung_ohne_bounds.py         # LSQ ohne Bounds
│  ├─ kalibrierung_mit_nachjustierung.py  # LSQ (Bounds) + globale Nachjustierung
│  ├─ einzelprofil_kalibrierung.py        # α anwenden auf alle Einzelprofile (Batch)
│  ├─ Sensitivitätsanalyse.py             # Bounds-Szenarien, Metriken, Wochenplots
│  └─ Validierung.py                      # Stichproben-Validierung, Plots + Kennzahlen
├─ cluster_basierte_kalibrierung_PAA/
│  └─ cluster_paa.py                      # Clustering auf PAA-Features (tslearn)
└─ data/
   └─ slp_h25_2023_hourly.csv             # Referenz-SLP (Spalte: Energy_kWh)
```

---

## Skripte im Detail

### Clustering

* **`cluster_basierte_kalibrierung/Cluster.py`**
  MiniBatchKMeans auf *voller* Jahreszeitreihe.
  **Ergebnis:** `cluster_labels_full_ts.csv`, `cluster_mean_full_ts.npy`

* **`cluster_basierte_kalibrierung_PAA/cluster_paa.py`**
  Clustering auf PAA-Features (Dim-Reduktion via `tslearn`).
  **Ergebnis:** `cluster_labels_paa.csv`, `cluster_centers_paa.npy`, `cluster_mean_full_paa.npy`

### Kalibrierung (α-Matrizen, Form K×T)

* **`least_squares_kalibrierung.py`**
  Zeitvariabler LSQ auf **aggregierter Skala**, Standard-Bounds `0.8…1.2`.
  **Ergebnis:** `calibration_matrix.npy`, `alphas_full_matrix.csv`

* **`kalibrierung_ohne_bounds.py`**
  LSQ **ohne** Bounds (unbounded).
  **Ergebnis:** `calibration_matrix_unbounded.npy`, `alphas_unbounded.csv`

* **`kalibrierung_mit_nachjustierung.py`**
  LSQ mit Bounds `0.8…1.2` **plus** globale Nachjustierung der Jahresenergie.
  **Ergebnis:** `calibration_matrix_bounded_adjusted.npy`, `alphas_bounded_adjusted.csv`

### Anwendung & Auswertung

* **`einzelprofil_kalibrierung.py`**
  Wendet `α_i(t)` je Cluster **batchweise** auf **alle** Einzelprofile an; Export in CSV-Chunks.
  **Ergebnis:** `calibrated_profiles_batches/calibrated_batch_###.csv`

* **`Sensitivitätsanalyse.py`**
  Bounds-Szenarien (`[0.8,1.2]` bis `[0.5,1.5]`), globale Nachjustierung, Kennzahlen & Wochenplots.
  **Ergebnis:** `alphas_bounds_adjust_only/*.npy`, `plots_bounds_adjust_only_kwh_phh/*.png`, `metrics_bounds_adjust_only_kwh_phh.csv`

* **`Validierung.py`**
  Stichproben-Validierung: mittlere Woche (SLP vs. unkalibriert vs. kalibriert), Jahres/Wochen-Metriken, Peaks.
  **Ergebnis:** `validation_plots/validation_mean_week_rep_#.png`, `validation_plots/validation_metrics_*.csv`

---

## Voraussetzungen

* **Python** ≥ 3.10
* **Pakete:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `tslearn`, `matplotlib`, `loguru`

```bash
pip install numpy pandas scipy scikit-learn tslearn matplotlib loguru
```

---

## Konfiguration & Datenquellen

**Interne Module:** `mwe_db_access.config`, `mwe_db_access.db`, `mwe_db_access.ssh`

* Stelle in `settings` den **SSH-Tunnel-Namen** und **DB-Zugang** ein; `register_schemas()` muss die relevanten Schemas registrieren.
* **DB-Tabelle:** `demand.iee_household_load_profiles` mit
  `id` *(int)* und `load_in_wh` *(Array/JSON mit 8760 Wh-Werten)*.
* **SLP-Datei:** `mwe_db_access/data/slp_h25_2023_hourly.csv` mit Spalte `Energy_kWh` *(8760 Zeilen)*.

> **Hinweis:** Einige Skripte enthalten **absolute Pfade** (z. B. `/home/emre/...`). Bitte am Skriptkopf anpassen oder zentral via Umgebungsvariablen/Config lösen.

---

## Standard-Parameter

| Bereich             | Wichtigste Parameter                                          | Default                |
| ------------------- | ------------------------------------------------------------- | ---------------------- |
| **Clustering**      | `K` (Clusteranzahl), `BATCH_SIZE`, `N_EPOCHS`, `RANDOM_STATE` | `10`, `500`, `2`, `42` |
| **Kalibrierung**    | Bounds `lb…ub`                                                | `0.8…1.2`              |
| **Batch-Anwendung** | `BATCH_SIZE` (Export kalibrierter Profile)                    | `1000`                 |

---

## Outputs

* **Cluster:** `cluster_labels_*.csv` *(id→cluster)*, `cluster_mean_full_*.npy` *(K×8760)*
* **Kalibrierung:** `calibration_matrix*.npy` *(K×8760)*, `alphas_*.csv` *(8760×K)*
* **Kalibrierte Profile:** `calibrated_profiles_batches/calibrated_batch_###.csv` *(Spalten: `id`, `load_in_wh_calibrated`)*
* **Plots & Metriken:** Wochenplots (PNG), Kennzahlen (CSV) inkl. RMSE/MAE/MAPE, Pearson, CV(RMSE), NMBE, Peaks, Energiesummen

---

## Tipps & Troubleshooting

* **Performance/RAM:** Batch-Größen bewusst wählen; `MiniBatchKMeans` arbeitet streaming-fähig.
* **SSH/DB:** Skripte öffnen selbst den Tunnel (`settings["ssh-tunnel"]["name"]`); gültige Keys/Agent sicherstellen.
* **Energie-Konsistenz:** SLP wird auf Zielenergie skaliert; die Variante *mit Nachjustierung* erzwingt exakte Jahressumme nach LSQ.
* **Pfad-Hygiene:** Absolute Pfade vermeiden/vereinheitlichen (z. B. via `BASE_DIR` oder `.env`).
