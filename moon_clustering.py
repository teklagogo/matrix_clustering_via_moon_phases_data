import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

# 1. Load Moon Dataset
FILE_PATH = r"moon_phases.txt"     
df = pd.read_csv(FILE_PATH, sep="\s+")

# Select numeric features
features = [
    'Phase','Age','Diam','Dist','RA','Dec',
    'Slon','Slat','Elon','Elat','AxisA'
]

X_raw = df[features].values

# Scale to [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2. Convert each observation into a 3×4 matrix
n = len(X_scaled)
X_matrices = np.zeros((n, 3, 4))

for i in range(n):
    flat = X_scaled[i]
    X_matrices[i, 0, :4] = flat[:4]
    X_matrices[i, 1, :4] = flat[4:8]
    X_matrices[i, 2, :3] = flat[8:11]   # last entry padded with 0

# Work with flattened representation for vectorized ops
X_flat = X_matrices.reshape(n, -1)  # shape (n, 12)

# 3.L∞ (max absolute entry difference) distance

def matrix_inf_norm(A_flat, B_flat):
    """A_flat, B_flat: 1D arrays of shape (12,) or 2D with shape (...,12).
       Returns L∞ distance(s) = max absolute entry difference.
    """
    return np.max(np.abs(A_flat - B_flat), axis=-1)

print("\n--- L∞ Norm Demonstration (vectorized, max abs entry) ---")
print("Example distance:", matrix_inf_norm(X_flat[0], X_flat[1]))

# 4. Custom L∞ K-Means (centroids = midpoint of min and max)
def kmeans_inf(X_flat, k, n_iter=100, random_state=42, tol=1e-6):
    np.random.seed(random_state)
    n, d = X_flat.shape
    # Initialize centers by sampling k points
    idx = np.random.choice(n, k, replace=False)
    centers = X_flat[idx].astype(float).copy()  # shape (k, d)

    for it in range(n_iter):
        # Vectorized assignment: distances shape (n, k)
        # dist[i, j] = L∞(X_flat[i], centers[j])
        diff = np.abs(X_flat[:, None, :] - centers[None, :, :])  # (n, k, d)
        dist = diff.max(axis=2)  # (n, k)
        labels = np.argmin(dist, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            members = X_flat[labels == j]
            if members.shape[0] == 0:
                # empty cluster -> reinitialize to a random point
                new_centers[j] = X_flat[np.random.choice(n)]
            else:
                # Chebyshev (L∞) minimizer per coordinate: midpoint (min+max)/2
                coord_min = members.min(axis=0)
                coord_max = members.max(axis=0)
                new_centers[j] = 0.5 * (coord_min + coord_max)

        # check convergence
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift <= tol:
            break

    return labels, centers

k = 6
start = time.time()
labels_km, centers_km_flat = kmeans_inf(X_flat, k=k, n_iter=200)
time_km = time.time() - start
centers_km = centers_km_flat.reshape(k, 3, 4)

# 5. Affinity Propagation using ∞ similarity matrix
sample_size = min(2000, n)
idx = np.random.choice(n, sample_size, replace=False)
X_sample_flat = X_flat[idx]

# Precompute similarity = -L∞ distance (precomputed affinity)
sim = np.zeros((sample_size, sample_size))
# vectorized computation of pairwise L∞ distances 
for i in range(sample_size):
    # broadcast: compute |X_sample_flat - X_sample_flat[i]| -> (sample_size, d)
    sim[:, i] = -np.max(np.abs(X_sample_flat - X_sample_flat[i]), axis=1)

start = time.time()
ap = AffinityPropagation(affinity='precomputed', damping=0.7, random_state=42)
ap.fit(sim)
time_ap = time.time() - start
labels_ap = ap.labels_

# 6. Silhouette using same L∞ metric
def silhouette_inf_fixed_from_flat(X_flat_local, labels_local):
    if len(set(labels_local)) <= 1:
        return -1.0
    n_local = len(X_flat_local)
    D_inf = np.zeros((n_local, n_local))
    for i in range(n_local):
        # compute distances to j>i and mirror
        dists = np.max(np.abs(X_flat_local[i] - X_flat_local), axis=1)
        D_inf[i, :] = dists
    sil = silhouette_samples(D_inf, labels_local, metric="precomputed")
    return float(np.mean(sil))

sil_km = silhouette_inf_fixed_from_flat(X_flat, labels_km)
sil_ap = silhouette_inf_fixed_from_flat(X_sample_flat, labels_ap)

# 7. Cluster sizes
uniq_km, count_km = np.unique(labels_km, return_counts=True)
uniq_ap, count_ap = np.unique(labels_ap, return_counts=True)

print("\n============= CLUSTER SUMMARY =============")

print("\n--- K-Means (true L∞) ---")
for u, c in zip(uniq_km, count_km):
    print(f"Cluster {u}: {c} samples")
print(f"Silhouette (∞): {sil_km:.4f}")
print(f"Runtime: {time_km:.3f} sec")

print("\n--- Affinity Propagation (L∞) ---")
for u, c in zip(uniq_ap, count_ap):
    print(f"Cluster {u}: {c} samples")
print(f"Silhouette (∞): {sil_ap:.4f}")
print(f"Runtime: {time_ap:.3f} sec")

# 8. Visualization (3D)
# Reduce each 3×4 matrix to 3 summary features: Phase, Age, Diam
X_summary = X_matrices.mean(axis=1)[:, :3]

fig = plt.figure(figsize=(16,6))

# ---- K-Means plot ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(
    X_summary[:,0],
    X_summary[:,1],
    X_summary[:,2],
    c=labels_km,
    cmap='tab10',
    alpha=0.6
)
ax1.set_title("K-Means Clustering (true L∞)")
ax1.set_xlabel("Phase")
ax1.set_ylabel("Age")
ax1.set_zlabel("Diam")

# ---- AP plot ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(
    X_summary[idx,0],
    X_summary[idx,1],
    X_summary[idx,2],
    c=labels_ap,
    cmap='tab10',
    alpha=0.6
)
ax2.set_title("Affinity Propagation (L∞)")
ax2.set_xlabel("Phase")
ax2.set_ylabel("Age")
ax2.set_zlabel("Diam")

plt.tight_layout()
plt.show()
