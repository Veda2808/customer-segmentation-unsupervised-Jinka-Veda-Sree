import pandas as pd

from src.clustering.kmeans import run_kmeans
from src.evaluation import evaluate_clustering

df = pd.read_csv('C:\Users\jinka vedasree\OneDrive\Desktop\customer-segmentation-unsupervised-Jinka veda sree\data\processed\clean_data.csv')

X = df.values

model, labels = run_kmeans(X)

sil, db = evaluate_clustering(X, labels)

print("Silhouette Score:", sil)
print("Davies Bouldin:", db)