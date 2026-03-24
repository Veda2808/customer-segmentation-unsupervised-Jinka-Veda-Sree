from sklearn.mixture import GaussianMixture

def run_gmm(X, k=4):

    model = GaussianMixture(n_components=k)

    labels = model.fit_predict(X)

    return model, labels