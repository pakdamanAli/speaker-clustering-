import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def cluster_embeddings(embeddings, use_cosine=True, n_clusters=None, threshold=0.5):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    if use_cosine:
        distance_matrix = 1 - cosine_similarity(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
            distance_threshold=threshold if n_clusters is None else None,
        )
        labels = clustering.fit_predict(distance_matrix)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="euclidean",
            linkage="ward",
            distance_threshold=threshold if n_clusters is None else None,
        )
        labels = clustering.fit_predict(embeddings)

    return labels
