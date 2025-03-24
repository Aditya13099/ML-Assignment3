# ML-Assignment3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def init_centroids(dataset, num_clusters):
    return random.sample(list(dataset), num_clusters)

def cluster_assignment(dataset, centers):
    return np.array([np.argmin([np.linalg.norm(point - center) for center in centers]) for point in dataset])

def recompute_centroids(dataset, labels, num_clusters):
    return np.array([dataset[labels == idx].mean(axis=0) if len(dataset[labels == idx]) > 0 else random.choice(dataset) for idx in range(num_clusters)])

def kmeans_clustering(dataset, num_clusters, max_iterations=100, threshold=1e-4):
    centers = np.array(init_centroids(dataset, num_clusters))
    for _ in range(max_iterations):
        labels = cluster_assignment(dataset, centers)
        updated_centers = recompute_centroids(dataset, labels, num_clusters)
        if np.linalg.norm(updated_centers - centers) < threshold:
            break
        centers = updated_centers
    return labels, centers

def visualize_clusters(dataset, labels, num_clusters):
    plt.figure(figsize=(6, 4))
    for idx in range(num_clusters):
        plt.scatter(*dataset[labels == idx].T, label=f"Group {idx}")
    plt.legend()
    plt.title(f"K-Means Clustering (k={num_clusters})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"kmeans_cluster_{num_clusters}.png")
    plt.close()

dataset = pd.read_csv("kmeans.csv")[['x1', 'x2']].values

for num_clusters in [2, 3]:
    cluster_labels, _ = kmeans_clustering(dataset, num_clusters)
    visualize_clusters(dataset, cluster_labels, num_clusters)
