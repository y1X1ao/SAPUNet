from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def compute_clustering_metrics(embedding, labels):
    metrics = {}
    try:
        metrics['Silhouette Score'] = silhouette_score(embedding, labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(embedding, labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(embedding, labels)
    except Exception as e:
        metrics['Error'] = str(e)
    return metrics
