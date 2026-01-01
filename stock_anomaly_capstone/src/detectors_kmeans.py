from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def fit_kmeans_train(X_train: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(X_train)
    return km

def kmeans_distance_to_centroid(km: KMeans, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # returns (cluster_id, distance)
    labels = km.predict(X)
    centers = km.cluster_centers_[labels]
    d = np.linalg.norm(X - centers, axis=1)
    return labels, d

def per_cluster_thresholds(train_labels: np.ndarray, train_d: np.ndarray, q: float) -> dict[int, float]:
    thr = {}
    for c in np.unique(train_labels):
        ds = train_d[train_labels == c]
        thr[int(c)] = float(np.percentile(ds, q))
    return thr

def flag_kmeans(labels: np.ndarray, dists: np.ndarray, thr_by_cluster: dict[int, float]) -> np.ndarray:
    out = np.zeros_like(dists, dtype=int)
    for i, (c, d) in enumerate(zip(labels, dists)):
        t = thr_by_cluster.get(int(c), np.inf)
        out[i] = int(d > t)
    return out
