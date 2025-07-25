import matplotlib.pyplot as plt
import numpy as np

"""
Hier wird das durchschnittliche Wochenprofil pro Cluster erstellt und geplottet
"""


# 1. Laden des Arrays (Form: (25, 8760))
cluster_data = np.load("cluster_mean_full_ts.npy")
n_clusters, n_timesteps = cluster_data.shape
print(f"Anzahl Cluster: {n_clusters}, Zeitpunkte pro Cluster: {n_timesteps}")

# 2. Parameter für Wochen:
hours_per_week = 7 * 24  # = 168
n_full_weeks = n_timesteps // hours_per_week  # 8760 // 168 = 52
n_used_hours = n_full_weeks * hours_per_week  # 52 * 168 = 8736

# 3. Für jedes Cluster: erst die letzten 24 Stunden abschneiden, dann in (52, 168) reshapen,
#    und entlang der Wochen-Matrix mitteln → ergibt pro Cluster einen Vektor der Länge 168.
mean_week_profiles = np.zeros((n_clusters, hours_per_week))
for k in range(n_clusters):
    # Rohdaten des k-ten Clusters
    ts = cluster_data[k, :n_used_hours]  # Länge = 8736
    # In Wochen chunken: Form wird (52, 168)
    ts_weeks = ts.reshape(n_full_weeks, hours_per_week)
    # Mittel über alle Wochen (Achse 0) → Array der Länge 168
    mean_week_profiles[k, :] = ts_weeks.mean(axis=0)

# 4. Plot aller 25 mittleren Wochenverläufe in einem Graphen
plt.figure(figsize=(12, 6))

# Falls du für jede Linie eine gut unterscheidbare Farbe möchtest, z.B. mit einem Hue-Map:
cmap = plt.get_cmap("hsv", n_clusters)

for k in range(n_clusters):
    plt.plot(
        np.arange(1, hours_per_week + 1),  # x-Achse: Stunde 1 bis 168
        mean_week_profiles[k, :],
        color=cmap(k),
        label=f"Cluster {k+1}",
    )

plt.title("Durchschnittlicher Wochenverlauf pro Cluster")
plt.xlabel("Stunde der Woche (1 – 168)")
plt.ylabel("Verbrauchswert")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="upper right", ncol=5, fontsize="small", framealpha=0.7)
plt.tight_layout()
plt.show()
