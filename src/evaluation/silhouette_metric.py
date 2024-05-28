from octis.evaluation_metrics.metrics import AbstractMetric
from sklearn.metrics import silhouette_score


class SilhouetteMetric(AbstractMetric):
    def __init__(self):
        """
        Initialize the Silhouette Metric
        """
        super().__init__()

    def info(self):
        return {"citation": {
            "silhouette": "Peter J. Rousseeuw (1987). \"Silhouettes: A graphical aid to the interpretation and validation of cluster analysis\". Journal of Computational and Applied Mathematics. 20: 53–65."},
            "name": "Silhouette Metric"}

    def score(self, model_output):
        """
        Retrieve the silhouette score for the model output

        Parameters
        ----------
        model_output : dictionary, output of the model
                       keys 'document_embeddings' and 'cluster_labels' required.

        Returns
        -------
        silhouette_score : silhouette score of the clustering
        """
        document_embeddings = model_output.get('document_embeddings')
        cluster_labels = model_output.get('cluster_labels')

        if document_embeddings is None or cluster_labels is None:
            raise ValueError("Model output must contain 'document_embeddings' and 'cluster_labels' keys")

        # Compute the silhouette score
        score = silhouette_score(document_embeddings, cluster_labels)

        print("Silhouette score: ", score)

        return score
