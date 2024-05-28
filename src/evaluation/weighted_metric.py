import numpy as np
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.metrics import AbstractMetric


class WeightedMetric(AbstractMetric):
    def __init__(self, texts=None, topk=10, processes=1, measure='c_npmi', diversity_weight=1, coherence_weight=1):
        """
        Initialize the weighted metric with log transformation

        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : top k words on which the metrics will be computed
        measure : (default 'c_npmi') measure to use for coherence.
        processes: number of processes for coherence
        diversity_weight: weight for the topic diversity score
        coherence_weight: weight for the coherence score
        """
        super().__init__()
        self.diversity_metric = TopicDiversity(topk=topk)
        self.coherence_metric = Coherence(texts=texts, topk=topk, processes=processes, measure=measure)
        self.diversity_weight = diversity_weight
        self.coherence_weight = coherence_weight

    def info(self):
        return {"citation": {"diversity": self.diversity_metric.info()["citation"],
                             "coherence": self.coherence_metric.info()["citation"]},
                "name": "Weighted Metric (Log Transformed Coherence and Topic Diversity)"}

    def score(self, model_output):
        """
        Retrieve the weighted score of the metrics with log transformation

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        weighted_score : weighted score combining log-transformed coherence and topic diversity
        """
        diversity_score = self.diversity_metric.score(model_output)
        coherence_score = self.coherence_metric.score(model_output)

        # Normalize scores to [0, 1] range
        max_diversity_score = 1.0  # Assuming the maximum possible diversity score is 1
        min_diversity_score = 0.0  # Assuming the minimum possible diversity score is 0
        max_coherence_score = 1.0  # Assuming the maximum possible coherence score is 1
        min_coherence_score = -1.0  # Assuming the minimum possible coherence score is -1 (for c_npmi)

        normalized_diversity_score = (diversity_score - min_diversity_score) / (
                max_diversity_score - min_diversity_score)
        normalized_coherence_score = (coherence_score - min_coherence_score) / (
                max_coherence_score - min_coherence_score)

        # Apply log transformation to avoid log(0) issues, add a small constant epsilon
        epsilon = 1e-10

        log_diversity_score = np.log(normalized_diversity_score + epsilon)
        log_coherence_score = np.log(normalized_coherence_score + epsilon)

        print("Coherence score: ", coherence_score, "Normalized coherence score: ", normalized_coherence_score,
              "Log coherence score: ", log_coherence_score)
        print("Diversity score: ", diversity_score, "Normalized diversity score: ", normalized_diversity_score,
              "Log diversity score: ", log_diversity_score)

        return [log_coherence_score, log_diversity_score]
