from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN

def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> DBSCAN:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model
