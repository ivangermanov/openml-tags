import ast

import bertopic as bt
import numpy as np
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from octis.models.model import AbstractModel
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class BERTopic(AbstractModel):
    def __init__(self, vectorizer_tokenizer=None, embeddings: np.ndarray = None, embedding_model=None, representation_model=None, verbose=False,
                 topk=10, use_partitions=False):
        super().__init__()
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.representation_model = representation_model
        self.vectorizer_tokenizer = vectorizer_tokenizer
        self.verbose = verbose
        self.topk = topk
        self.use_partitions = use_partitions
        self.model = None

    def train_model(self, dataset, hyperparameters, top_words=10):
        if hyperparameters is None:
            hyperparameters = {}

        super().set_hyperparameters(**hyperparameters)
        print("Current params: ", hyperparameters)

        data = dataset.get_corpus()
        data = [" ".join(words) for words in data]
        umap_model = UMAP(n_neighbors=hyperparameters.get("umap_n_neighbors", 15),
                          n_components=hyperparameters.get("umap_n_components", 5),
                          min_dist=hyperparameters.get("umap_min_dist", 0.0),
                          metric=hyperparameters.get("umap_metric", 'cosine'))

        hdbscan_model = HDBSCAN(min_cluster_size=hyperparameters.get("hdbscan_min_cluster_size", 5),
                                metric=hyperparameters.get("hdbscan_metric", 'euclidean'),
                                cluster_selection_method=hyperparameters.get("hdbscan_cluster_selection_method", 'eom'),
                                prediction_data=True)

        vectorizer_tokenizer = (self.vectorizer_tokenizer if ast.literal_eval(
            hyperparameters.get("vectorizer_tokenizer", "False")) else None)
        vectorizer_model = CountVectorizer(
            ngram_range=ast.literal_eval(hyperparameters.get("vectorizer_ngram_range", (1, 1))),
            stop_words=hyperparameters.get("vectorizer_stop_words", "english"), tokenizer=vectorizer_tokenizer)

        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=ast.literal_eval(hyperparameters.get("ctfidf_reduce_frequent_words", False)))

        self.model = bt.BERTopic(language=hyperparameters.get("language", "english"),
                                 top_n_words=hyperparameters.get("top_n_words", 10),
                                 n_gram_range=hyperparameters.get("n_gram_range", (1, 1)),
                                 min_topic_size=hyperparameters.get("min_topic_size", 10),
                                 nr_topics=int(hyperparameters.get("nr_topics")) if hyperparameters.get("nr_topics",
                                                                                                        None) else None,
                                 low_memory=hyperparameters.get("low_memory", False),
                                 calculate_probabilities=hyperparameters.get("calculate_probabilities", False),
                                 seed_topic_list=hyperparameters.get("seed_topic_list", None),
                                 zeroshot_topic_list=hyperparameters.get("zeroshot_topic_list", None),
                                 zeroshot_min_similarity=hyperparameters.get("zeroshot_min_similarity", 0.9),
                                 embedding_model=self.embedding_model, umap_model=umap_model,
                                 hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
                                 ctfidf_model=ctfidf_model,
                                 representation_model=self.representation_model,
                                 verbose=self.verbose)

        topics, probs = self.model.fit_transform(data, self.embeddings)
        
        if hyperparameters.get("outliers_strategy", "none") != "none":
            try:
                topics = self.model.reduce_outliers(data, topics, probabilities=probs, embeddings=self.embeddings, strategy=hyperparameters.get("outliers_strategy", "probabilities"))
                self.model.update_topics(data, topics=topics)
            except Exception as e:
                print('Error in outliers reduction', e)
                hyperparameters['outliers_strategy'] = 'none'
        
        all_words = [word for words in dataset.get_corpus() for word in words]
        bertopic_topics = [
            [vals[0] if vals[0] in all_words else all_words[0] for vals in self.model.get_topic(i)[:self.topk]] for i in
            range(len(set(topics)) - 1)]

        print("Topics: ", bertopic_topics)

        umap_embeddings = self.model.umap_model.transform(self.embeddings)
        indices = [index for index, topic in enumerate(topics) if topic != -1]
        document_embeddings = umap_embeddings[np.array(indices)].tolist()
        cluster_labels = [topic for index, topic in enumerate(topics) if topic != -1]

        output_tm = {"topics": bertopic_topics, "document_embeddings": document_embeddings,
                     "cluster_labels": cluster_labels}

        return output_tm
