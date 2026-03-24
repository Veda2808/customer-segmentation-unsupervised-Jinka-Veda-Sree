from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

def evaluate_clustering(X, labels):

    sil = silhouette_score(X, labels)

    db = davies_bouldin_score(X, labels)

    return sil, db