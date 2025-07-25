import matplotlib.pyplot as plt
import numpy as np

"""
Hier wird das Jahresprofil jedes Clusters gebildet und geplottet
"""

# 1. Laden des Arrays
cluster_data = np.load(
    "/home/emre/MA/PycharmProjects/cluster_kalibrierung/mwe_db_access/data/cluster_mean_full_ts.npy"
)  # Form: (n_clusters, 8760)

# 2. Prüfen, wie viele Cluster es sind
n_clusters, n_timesteps = cluster_data.shape
print(f"Anzahl Cluster: {n_clusters}, Zeitpunkte pro Cluster: {n_timesteps}")

# 3. Plot pro Cluster
for k in range(n_clusters):
    plt.figure(figsize=(12, 4))
    plt.plot(cluster_data[k, :])
    plt.title(f"Cluster {k+1}: Jahreszeitlicher Verlauf (8760 Stunden)")
    plt.xlabel("Stunde des Jahres (1–8760)")
    plt.ylabel("Verbrauchswert")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
