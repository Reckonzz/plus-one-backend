import numpy as np

def clustering_model(inputs, type="kmeans"): 
    inputs = np.array(inputs)
    n_clusters = 3
    clusters = list(map(lambda x: x.tolist(),np.array_split(inputs, n_clusters)))
    labels = [f"label_{i+1}" for i in range(n_clusters)]
    print(clusters)
    print(labels ) 
    return clusters, labels

