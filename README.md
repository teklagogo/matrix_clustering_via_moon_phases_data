# 🌙 Matrix-Based Moon Observation Clustering

> **NLA with ML | CP#1.1** — Kutaisi International University  
> *Tekla Mamageishvili*

---

## Overview

This project applies matrix-based clustering techniques to moon phase observation data. Each observation is represented as a **3×4 matrix**, and all distances/similarities are computed using the **L∞ (infinity norm / Chebyshev distance)** — the maximum absolute difference between matrix entries.

Two algorithms are implemented and compared:
- **K-Means** with a custom L∞ distance and Chebyshev midpoint update rule
- **Affinity Propagation** with L∞-based pairwise similarities

---

## Matrix Representation

Each moon observation is encoded as a 3×4 matrix:

```
Xi = | Phase   Age    Diam   Dist |
     | RA      Dec    Slon   Slat |
     | Elon    Elat   AxisA  0    |
```

The infinity norm between two observations A and B:

```
‖A − B‖∞ = max |aᵢⱼ − bᵢⱼ|
             i,j
```

---

## Methods

### 1. K-Means Clustering (L∞)

**Problem Statement:**
- Normalize 11 features to [0, 1]
- Reshape each observation into a 3×4 matrix
- Run K-Means with `k = 6` using L∞ distance
- Evaluate with silhouette scores under L∞

**Pipeline:**
1. Load and normalize the dataset
2. Convert each observation to a 3×4 matrix
3. Flatten matrices to 12-dimensional vectors for computation
4. Run custom K-Means:
   - Distance = L∞ (max absolute difference)
   - Cluster center update = midpoint of componentwise min and max (Chebyshev midpoint rule)
5. Compute silhouette scores using L∞

**Results:**

| Cluster | Samples |
|---------|---------|
| 0       | 1611    |
| 1       | 1340    |
| 2       | 1532    |
| 3       | 1548    |
| 4       | 785     |
| 5       | 1944    |

- **Silhouette Score (L∞):** `0.1431`
- **Runtime:** `0.119 sec`

---

### 2. Affinity Propagation (L∞)

**Pipeline:**
1. Randomly sample 2000 observations
2. Compute the 2000×2000 L∞ distance matrix
3. Convert to similarity matrix: `S = −D∞`
4. Fit Affinity Propagation with `damping = 0.7`
5. Compute silhouette using L∞ metric

**Results:**

| Metric              | Value   |
|---------------------|---------|
| Number of Clusters  | 139     |
| Silhouette (L∞)     | 0.5107  |
| Runtime             | 3.451 sec |

---

## Comparison

| Algorithm                    | # Clusters | Silhouette (L∞) | Runtime (sec) |
|-----------------------------|------------|-----------------|---------------|
| K-Means (L∞)                | 6          | 0.1431          | 0.119         |
| Affinity Propagation (L∞)   | 139        | 0.5107          | 3.451         |

**Key takeaway:** K-Means captures broad structural groupings quickly, while Affinity Propagation discovers finer-grained patterns through exemplar-based clustering — at the cost of higher runtime and granularity.

---

## Conclusion

Using L∞ metrics provides a geometrically consistent framework across both algorithms. The infinity norm is well-suited for structured moon observations because it emphasizes the most extreme feature deviations rather than averaging across all features (as Euclidean distance does).

---

## Dependencies

- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [scikit-learn](https://scikit-learn.org)
- [Matplotlib](https://matplotlib.org)

---

## References

1. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
2. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering, 9(3), 90–95.
3. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proc. 9th Python in Science Conf., 51–56.
4. van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). *The NumPy Array.* Computing in Science & Engineering, 13(2), 22–30.
