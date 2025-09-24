# Cluster-Kalibrierung von Haushaltslasten

Ziel: Viele Haushalts-Lastprofile (je 8760 Stundenwerte) so anpassen, dass Form und Energie besser zu **SLP H25** passen.

Kurzablauf:
**Clustern → stündliche Skalierungsfaktoren je Cluster → (optional) Jahresenergie nachjustieren → auf alle Profile anwenden → prüfen.**

---

## Wichtiger Hinweis zum Datenzugang

Ein Teil der Skripte liest die Profile direkt aus einer Datenbank, die nur über das **Reiner-Lemoine-Institut** erreichbar ist (Server/SSH-Tunnel, pgAdmin, **egon-data**).
Ohne diesen Zugang laufen die betreffenden Skripte nicht. Besonders betroffen: **`Cluster.py`** (Clustering aus DB) und **`einzelprofil_kalibrierung.py`** (Batch-Abruf der Originalprofile).

---

## Voraussetzungen

* **Python** >= 3.10
* Pakete:

  ```bash
  pip install numpy pandas scipy scikit-learn tslearn matplotlib loguru
  ```
* Bei RLI/egon-data: gültige SSH/DB-Einstellungen in `mwe_db_access.config.settings`

---

## Projektstruktur (kurz)

```
mwe_db_access/
├─ cluster_basierte_kalibrierung/
│  ├─ Cluster.py                       # Clustering auf vollen 8760h-Profilen (DB)
│  ├─ least_squares_kalibrierung.py    # Kalibrierung mit Bounds
│  ├─ kalibrierung_ohne_bounds.py      # Kalibrierung ohne Bounds
│  ├─ kalibrierung_mit_nachjustierung.py  # Bounds + globale Jahresenergie-Anpassung
│  ├─ einzelprofil_kalibrierung.py     # Alpha auf alle Profile anwenden (DB, Batch)
│  ├─ Sensitivitätsanalyse.py          # Bounds-Szenarien, Kennzahlen, Plots
│  └─ Validierung.py                   # Stichprobenplots + Metriken
├─ cluster_basierte_kalibrierung_PAA/
│  └─ cluster_paa.py                   # Clustering mit PAA-Features (optional)
└─ data/
   └─ slp_h25_2023_hourly.csv          # Referenz-SLP (Spalte: Energy_kWh)
```

---

## Typischer Ablauf (mit RLI/egon-data)

```bash
# 1) Cluster bilden (liest aus DB)
python mwe_db_access/cluster_basierte_kalibrierung/Cluster.py

# 2) Kalibrierung (empfohlen)
python mwe_db_access/cluster_basierte_kalibrierung/kalibrierung_mit_nachjustierung.py
# Alternativ:
# python mwe_db_access/cluster_basierte_kalibrierung/least_squares_kalibrierung.py
# python mwe_db_access/cluster_basierte_kalibrierung/kalibrierung_ohne_bounds.py

# 3) Alpha auf alle Einzelprofile anwenden (Batch-Export; liest Originalprofile aus DB)
python mwe_db_access/cluster_basierte_kalibrierung/einzelprofil_kalibrierung.py

# 4) Optional: Sensitivität
python mwe_db_access/cluster_basierte_kalibrierung/Sensitivitätsanalyse.py

# 5) Validierung (Stichprobe/Plots)
python mwe_db_access/cluster_basierte_kalibrierung/Validierung.py
```

**Standard-Parameter (meist okay so):**

* K = 10 Cluster
* BATCH\_SIZE = 500 (Clustering) / 1000 (Anwenden)
* Bounds bei der Kalibrierung = 0.8 … 1.2

---

## Was machen die wichtigsten Skripte?

| Skript                               | Kurzbeschreibung                                                                    | Wichtige Outputs                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `Cluster.py`                         | MiniBatchKMeans auf vollen 8760h-Profilen (aus DB)                                  | `cluster_labels_full_ts.csv`, `cluster_mean_full_ts.npy`                         |
| `cluster_paa.py`                     | Clustering auf PAA-Features (Dim-Reduktion)                                         | `cluster_labels_paa.csv`, `cluster_centers_paa.npy`, `cluster_mean_full_paa.npy` |
| `kalibrierung_mit_nachjustierung.py` | Stündliche Skalierung je Cluster (Bounds 0.8–1.2) + globale Jahresenergie-Anpassung | `calibration_matrix_bounded_adjusted.npy`, `alphas_bounded_adjusted.csv`         |
| `least_squares_kalibrierung.py`      | Wie oben, ohne finale Nachjustierung                                                | `calibration_matrix.npy`, `alphas_full_matrix.csv`                               |
| `kalibrierung_ohne_bounds.py`        | LSQ ohne Grenzen (eher zum Testen)                                                  | `calibration_matrix_unbounded.npy`, `alphas_unbounded.csv`                       |
| `einzelprofil_kalibrierung.py`       | Wendet die Alpha-Matrix auf **alle** Profile an (DB, Batch)                         | `calibrated_profiles_batches/calibrated_batch_###.csv`                           |
| `Sensitivitätsanalyse.py`            | Testet verschiedene Bounds, schreibt Kennzahlen + Plots                             | `metrics_*.csv`, `plots_*.png`                                                   |
| `Validierung.py`                     | Stichproben-Plots und Metriken (Jahr & Woche, Peaks)                                | `validation_plots/*.png`, `validation_plots/*.csv`                               |

---

## Outputs (Erwartung)

* **Cluster:** `cluster_labels_*.csv` (id -> cluster), `cluster_mean_full_*.npy` (K x 8760)
* **Kalibrierung:** `calibration_matrix*.npy` (K x 8760), `alphas_*.csv` (8760 x K)
* **Kalibrierte Profile:** `calibrated_profiles_batches/calibrated_batch_###.csv` (`id`, `load_in_wh_calibrated`)
* **Plots & Metriken:** PNG-Plots und CSV-Kennzahlen (RMSE/MAE/MAPE, Korrelation, CV(RMSE), NMBE, Peaks, Energiesummen)

---

## Konfiguration & Pfade

* In manchen Skripten stehen **absolute Pfade** (z. B. `/home/emre/...`). Bitte am Skriptkopf anpassen.
* DB/SSH-Zugänge (falls genutzt) kommen aus `mwe_db_access.config.settings`; außerdem wird `register_schemas()` aufgerufen.

---

## Häufige Stolpersteine

* **Kein DB-Zugang** → Die DB-basierten Skripte können dann nicht laufen.
* **RAM/Performance** → `BATCH_SIZE` kleiner wählen (MiniBatchKMeans ist streaming-fähig).
* **Energiesumme passt nicht** → Skript mit **Nachjustierung** verwenden (`kalibrierung_mit_nachjustierung.py`).
* **SLP fehlt** → `data/slp_h25_2023_hourly.csv` bereitstellen (Spalte `Energy_kWh`).

---

## Ein Satz zur Methode

Die Profile werden in **K** Gruppen geclustert. Für jede Stunde werden je Cluster **Skalierungsfaktoren** bestimmt, damit die **Summe** aller Profile dem SLP zeitlich besser folgt (mit Grenzen, damit die Form pro Cluster realistisch bleibt). Optional wird am Ende die **Jahresenergie** global korrigiert.


ChatGPT diente bei der Entwicklung des Codes als Hilfsmittel. Alle Ergebnisse wurden eigenständig kontrolliert und bearbeitet.


