"""
SLP H25 → stündliches Energieprofil (kWh)

Dieses Skript liest das BDEW-Lastprofil H25 (15-Minuten-Werte in kWh)
aus einer Excel-Datei ein und wandelt es in ein durchgängiges,
stündliches Jahresprofil (8760 Werte) um. Zusätzlich werden Monats-
Maxima und Gesamtkennzahlen (Jahresenergie, Spitzenstunde, Mittelwert)
berechnet und ausgegeben. Am Ende wird das Ergebnis als CSV und Plot
gespeichert/angezeigt.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from holidays import CountryHoliday


def slp_h25_to_hourly(
    filepath: str, sheet_name: str = "H25", year: int = 2023, country: str = "DE"
) -> pd.Series:
    """
    Liest das SLP H25 Excel (15-Minuten-Profile in kWh) ein und gibt eine
    pd.Series mit 8760 stündlichen Energie-Werten (kWh) für das ganze Jahr
    zurück.

    Index:  pd.DatetimeIndex von 2023-01-01 00:00 bis 2023-12-31 23:00 (stündlich)
    Werte:  Energie in kWh pro Stunde
    """
    # 1) Excel einlesen (MultiIndex: Monat, Tagestyp)
    df = pd.read_excel(
        filepath, sheet_name=sheet_name, header=[0, 1], index_col=0
    ).dropna(how="all")
    df.columns.names = ["month", "daytype"]

    # 2) Monatsnamen → Monatszahl
    month_map = {
        mon: pd.to_datetime(mon, format="%B").month for mon in df.columns.levels[0]
    }

    # 3) Profile dict: (Monat, Tagestyp) → Array (96 Viertelstunden-Werte in kWh)
    profiles = {
        (month_map[mon], dt): df[mon, dt].values
        for mon in df.columns.levels[0]
        for dt in df[mon].columns
    }

    # 4) Alle Kalendertage des Jahres
    days = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="D", closed="left")

    # 5) Feiertage (DE)
    hols = CountryHoliday(country, years=[year])

    # 6) Für jeden Tag das 96-Werte-Profil auswählen
    quarter_list = []
    for d in days:
        wd = d.weekday()  # Mo=0 … So=6
        if wd == 5:
            typ = "SA"
        elif wd == 6 or d in hols:
            typ = "FT"
        else:
            typ = "WT"
        quarter_list.append(profiles[(d.month, typ)])

    # 7) Viertelstunden (kWh) → Stundenenergie (kWh)
    all_q = np.concatenate(quarter_list)  # Array-Länge: 365×96
    hourly = all_q.reshape(-1, 4).sum(axis=1)  # Array-Länge: 8760

    # 8) Series mit DatetimeIndex stündlich
    idx = pd.date_range(f"{year}-01-01", periods=hourly.size, freq="H")
    return pd.Series(hourly, index=idx, name="Energy_kWh")


if __name__ == "__main__":
    # Pfad zur Excel-Datei anpassen
    excel_path = "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/bdew_h_25.xlsx"

    # in stündliches Profil umwandeln
    load_series = slp_h25_to_hourly(excel_path, sheet_name="H25", year=2023)

    # CSV schreiben
    csv_path = "../data/slp_h25_2023_hourly.csv"
    load_series.to_csv(csv_path, header=True, index_label="Timestamp")
    print(f"CSV geschrieben: {csv_path}\n")

    # Plot erstellen
    ax = load_series.plot(figsize=(15, 4), legend=False)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Energie pro Stunde (kWh)")
    ax.set_title("SLP H25 stündliches Jahresprofil 2023 (kWh)")
    plt.tight_layout()
    plt.show()

    # --- Kennzahlen berechnen ---
    # Monats-Maxima
    monthly_max = load_series.resample("M").max()
    print("Monatliche Spitzenwerte (kWh):")
    print(monthly_max.to_string(), "\n")

    # Jahresgesamtenergie
    total_energy_kwh = load_series.sum()
    # Spitzenstunde
    peak_energy_kwh = load_series.max()
    peak_time = load_series.idxmax()
    # Mittelwert pro Stunde
    average_hourly_kwh = load_series.mean()

    print(f"Jahresenergiebedarf:      {total_energy_kwh:.0f} kWh")
    print(f"Maximale Stundenenergie:   {peak_energy_kwh:.2f} kWh um {peak_time}")
    print(f"Durchschnitt pro Stunde:   {average_hourly_kwh:.2f} kWh")
