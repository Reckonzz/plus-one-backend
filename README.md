# plus-one-backend

This is the flask app that we call 'GET' request to predict clusters and its labels ( in the form of summary).
- OpenAI summarizer (GPT3) -> for cluster label
- SBert -> for embedding
- Kmeans -> for clustering 
- silhouette score -> for choosing best k
