from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd

def run_dbscan(X, eps=0.5, min_samples=5):

    model = DBSCAN(eps=eps, min_samples=min_samples)

    labels = model.fit_predict(X)

    return model, labels
    