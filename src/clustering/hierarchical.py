from sklearn.cluster import AgglomerativeClustering
import numpy as np

def run_hierarchical(X, k=4):

    # Take only 1000 samples to avoid memory error
    sample_size = 1000
    
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    model = AgglomerativeClustering(n_clusters=k)

    labels = model.fit_predict(X_sample)

    return model, labels, X_sample