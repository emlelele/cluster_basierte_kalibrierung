import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MultipleLocator

# ----------------------------
# KONFIGURATION UND PARAMETER
# ----------------------------
K = 10  # Anzahl der Cluster
T = 8760  # Stunden pro Jahr
HOURS_PER_WEEK = 168  # Stunden pro Woche
YLIM = (0, 2100)  # feste Y-Achse (Wh) für alle Plots

# ID-Bereiche → Haushaltskategorie
CATEGORY_RANGES = {
    "OO": (1, 800),
    "OR": (801, 1700),
    "P1": (1701, 11700),
    "P2": (11701, 21700),
    "P3": (21701, 27200),
    "PO": (27201, 44200),
    "PR": (44201, 55200),
    "SK": (55201, 61600),
    "SO": (61601, 85600),
    "SR": (85601, 100000),
}

# ----------------------------
# 1) CLUSTER-LABELS LADEN UND KATEGORIE ZUORDNEN
# ----------------------------
df = pd.read_csv("cluster_labels_paa.csv")


def assign_category(pid: int) -> str:
    for cat, (low, high) in CATEGORY_RANGES.items():
        if low <= pid <= high:
            return cat
    return "Unknown"


df["category"] = df["id"].apply(assign_category)
df.to_csv("cluster_labels_paa_with_cat.csv", index=False)
print("→ Gespeichert: cluster_labels_paa_with_cat.csv mit Spalte 'category'")

# ----------------------------
# 2) MITTELPROFILE LADEN
# ----------------------------
cluster_mean_full = np.load("cluster_mean_full_paa.npy")

# ----------------------------
# 3) DURCHSCHNITTLICHE WOCHE PRO CLUSTER BERECHNEN & PLOTTEN
# ----------------------------
for idx in range(K):
    cluster_num = idx + 1
    mean_year = cluster_mean_full[idx]

    # a) Durchschnittliche Woche berechnen
    weekly_avg = np.array(
        [mean_year[h::HOURS_PER_WEEK].mean() for h in range(HOURS_PER_WEEK)]
    )

    # b) Häufigkeit jeder Haushaltskategorie in diesem Cluster
    subset = df[df["cluster"] == idx]
    cnts = subset["category"].value_counts().to_dict()

    # Infobox-Text mit Header
    info_text = "Haushaltstypen:\n" + "\n".join(
        f"{cat}: {cnt}" for cat, cnt in cnts.items()
    )

    # c) Plot erzeugen
    fig, ax = plt.subplots(figsize=(10, 5))
    # Stunden von 1…168
    hours = np.arange(1, HOURS_PER_WEEK + 1)
    ax.plot(hours, weekly_avg, label=f"Cluster {cluster_num}", color="tab:blue")

    # Original-Titel und Beschriftungen
    ax.set_title(f"Durchschnittliche Woche mit PAA – Cluster {cluster_num}")
    ax.set_xlabel("Stunde der Woche (1–168)")
    ax.set_ylabel("Verbrauch (Wh)")
    ax.set_ylim(*YLIM)

    # Ticks und Raster
    ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.grid(which="major", linestyle="--", linewidth=0.5, color="gray", alpha=0.7)

    # Legend für die Kurve
    ax.legend(loc="upper left", framealpha=0.9)

    # Infobox
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#f0f8ff", alpha=0.9)
    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
    )

    plt.tight_layout()
    out_filename = f"avg_week_cluster_{cluster_num}.png"
    plt.savefig(out_filename, dpi=150)
    plt.close(fig)
    print(f"→ Plot gespeichert: {out_filename}")

print("Alle Plots erstellt.")
