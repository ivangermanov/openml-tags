from collections import Counter

import numpy as np
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.metrics import AbstractMetric


class WeightedMetric(AbstractMetric):
    def __init__(self, texts=None, topk=10, processes=1, measure='c_npmi', diversity_weight=1, coherence_weight=1,
                 repetition_weight=0.2):
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
        repetition_weight: weight for the term repetition penalty
        """
        super().__init__()
        self.diversity_metric = TopicDiversity(topk=topk)
        self.coherence_metric = Coherence(texts=texts, topk=topk, processes=processes, measure=measure)
        self.diversity_weight = diversity_weight
        self.coherence_weight = coherence_weight
        self.repetition_weight = repetition_weight

    def info(self):
        return {"citation": {"diversity": self.diversity_metric.info()["citation"],
                             "coherence": self.coherence_metric.info()["citation"]},
                "name": "Weighted Metric (Log Transformed Coherence and Topic Diversity)"}

    @staticmethod
    def calculate_term_repetition_penalty(topics):
        penalties = []
        for topic in topics:
            term_counts = Counter(topic)
            penalty = sum(count - 1 for count in term_counts.values() if count > 1)
            penalties.append(penalty)
        return sum(penalties) / len(topics)

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
        topics = model_output['topics']

        diversity_score = self.diversity_metric.score(model_output)
        coherence_score = self.coherence_metric.score(model_output)
        repetition_penalty = self.calculate_term_repetition_penalty(topics)

        # Normalize scores to [0, 1] range
        max_diversity_score = 1.0  # Assuming the maximum possible diversity score is 1
        min_diversity_score = 0.0  # Assuming the minimum possible diversity score is 0
        max_coherence_score = 1.0  # Assuming the maximum possible coherence score is 1
        min_coherence_score = -1.0  # Assuming the minimum possible coherence score is -1 (for c_npmi)

        normalized_diversity_score = (diversity_score - min_diversity_score) / (
                max_diversity_score - min_diversity_score)
        normalized_coherence_score = (coherence_score - min_coherence_score) / (
                max_coherence_score - min_coherence_score)

        # Normalize the term repetition penalty (assuming max possible penalty is len(topics))
        max_repetition_penalty = len(topics)
        min_repetition_penalty = 0

        normalized_repetition_penalty = (repetition_penalty - min_repetition_penalty) / (
                max_repetition_penalty - min_repetition_penalty)

        # Apply log transformation to avoid log(0) issues, add a small constant epsilon
        epsilon = 1e-10

        log_diversity_score = np.log(normalized_diversity_score + epsilon)
        log_coherence_score = np.log(normalized_coherence_score + epsilon)
        log_repetition_penalty = np.log(normalized_repetition_penalty + epsilon)

        print("coherence", coherence_score, normalized_coherence_score, log_coherence_score)
        print("diversity", diversity_score, normalized_diversity_score, log_diversity_score)
        print("repetition penalty", repetition_penalty, normalized_repetition_penalty, log_repetition_penalty)

        weighted_score = self.diversity_weight * log_diversity_score + self.coherence_weight * log_coherence_score + self.repetition_weight * log_repetition_penalty

        print("weighted_score", weighted_score)
        return [log_coherence_score, log_diversity_score]
        # return weighted_score
