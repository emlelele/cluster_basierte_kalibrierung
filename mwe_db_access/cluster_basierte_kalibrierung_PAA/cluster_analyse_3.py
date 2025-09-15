import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # a) Durchschnittliche Woche berechnen (1..168)
    weekly_avg = np.array(
        [mean_year[h::HOURS_PER_WEEK].mean() for h in range(HOURS_PER_WEEK)]
    )

    # b) Häufigkeit jeder Haushaltskategorie in diesem Cluster
    subset = df[df["cluster"] == idx]
    cnts = subset["category"].value_counts()

    info_text = "Haushaltstypen:\n" + "\n".join(
        f"{cat}: {cnt}" for cat, cnt in cnts.items()
    )

    # c) Plot im exakt gleichen Stil wie zuvor
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = np.arange(1, HOURS_PER_WEEK + 1)  # 1..168
    ax.plot(hours, weekly_avg, linewidth=2, label=f"Cluster {cluster_num}")

    ax.set_title(f"Durchschnittliche Woche – Cluster mit PAA {cluster_num}")
    ax.set_xlabel("Stunde der Woche (1–168)")
    ax.set_ylabel("Last (Wh)")
    ax.set_ylim(*YLIM)

    # 24h-Sequenzen wie zuvor: Major bei 1, 24, 48, …; Minor alle 6h
    ax.set_xticks([1] + list(range(24, HOURS_PER_WEEK + 1, 24)))
    ax.set_xticks(range(6, HOURS_PER_WEEK + 1, 6), minor=True)

    # Grid-Stil wie zuvor
    ax.grid(True, linestyle="--", alpha=0.3)

    # Legend wie zuvor
    ax.legend(loc="upper left", frameon=True)

    # Hellblaue, größere Info-Box oben rechts (gleicher Stil wie angepasst)
    ax.text(
        0.985, 0.985, info_text,
        transform=ax.transAxes,
        fontsize=14,
        linespacing=1.4,
        va="top", ha="right",
        family="DejaVu Sans Mono",
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="lightblue",
            edgecolor="black",
            linewidth=1.3,
            alpha=0.9
        )
    )

    plt.tight_layout()
    out_filename = f"avg_week_cluster_{cluster_num}.png"
    plt.savefig(out_filename, dpi=150)
    plt.close(fig)
    print(f"→ Plot gespeichert: {out_filename}")

print("Alle Plots erstellt.")
