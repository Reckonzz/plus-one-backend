import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')

def clustering_model(sticky_notes, type="kmeans"): 
    embeddings = model_encoder.encode(sticky_notes, convert_to_numpy=True)

    kmeans_kwargs = {
        "init": "random",
        "n_init": 20,
        "max_iter": 300,
        "random_state": 42,
    }

    # A list holds the silhouette coefficients for each k

    best_k = 1 
    best_score = 0 
    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, len(sticky_notes)):
        print(k)
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        if score > best_score:
            best_k = k
            best_score = score

    kmeans = KMeans(n_clusters=best_k,  **kmeans_kwargs)

    kmeans.fit(embeddings)

    group ={}

    for i in range(len(sticky_notes)):
        if kmeans.labels_[i] in group:
            group[kmeans.labels_[i]].append(sticky_notes[i])
        else:
            group[kmeans.labels_[i]] = [sticky_notes[i]]
    # n_clusters = 3
    # clusters = list(map(lambda x: x.tolist(),np.array_split(inputs, n_clusters)))
    # labels = [f"label_{i+1}" for i in range(n_clusters)]
    # print(clusters)
    # print(labels ) 
    return list(group.values()), [i for i in range(len(group.values()))]

