from typing import Mapping, List, Tuple, Any, Callable, Union

import pandas as pd
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document
from scipy.sparse import csr_matrix
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.base import Pipeline

DEFAULT_PROMPT = """
I have topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
"""


class ZeroShotClassification(BaseRepresentation):
    """ Zero-shot Classification on topic keywords with candidate labels

    Arguments:
        candidate_topics: A list of labels to assign to the topics if they
                          exceed `min_prob`
        model: A transformers pipeline that should be initialized as
               "zero-shot-classification". For example,
               `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")`
        pipeline_kwargs: Kwargs that you can pass to the transformers.pipeline
                         when it is called. NOTE: Use `{"multi_label": True}`
                         to extract multiple labels for each topic.
        min_prob: The minimum probability to assign a candidate label to a topic

    Usage:

    ```python
    # Create your representation model
    candidate_topics = ["space and nasa", "bicycles", "sports"]
    representation_model = ZeroShotClassification(candidate_topics, model="facebook/bart-large-mnli")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """

    def __init__(self, candidate_topics: List[str], model: str = "facebook/bart-large-mnli",
                 pipeline_kwargs: Mapping[str, Any] = {}, min_prob: float = 0.5, prompt: str = None, nr_docs: int = 8,
                 diversity: float = None, doc_length: int = None, tokenizer: Union[str, Callable] = None):
        self.candidate_topics = candidate_topics
        if isinstance(model, str):
            self.model = pipeline("zero-shot-classification", model=model)
        elif isinstance(model, Pipeline):
            self.model = model
        else:
            raise ValueError("Make sure that the HF model that you"
                             "pass is either a string referring to a"
                             "HF model or a `transformers.pipeline` object.")
        self.pipeline_kwargs = pipeline_kwargs
        self.min_prob = min_prob
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer

        self.prompts_ = []

    def extract_topics(self, topic_model, documents: pd.DataFrame, c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: A BERTopic model
            documents: The original documents
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Classify topics
        # topic_descriptions = [" ".join(list(zip(*topics[topic]))[0]) for topic in topics.keys()]
        # classifications = self.model(topic_descriptions, self.candidate_topics, **self.pipeline_kwargs)
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(c_tf_idf, documents, topics, 500,
                                                                               self.nr_docs, self.diversity)

        # Extract labels
        updated_topics = {}
        # for topic, classification in zip(topics.keys(), classifications):
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            topic_description = topics[topic]
            # Prepare prompt
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)
            classification = self.model(prompt, self.candidate_topics, **self.pipeline_kwargs)

            # Multi-label assignment
            if self.pipeline_kwargs.get("multi_label"):
                topic_description = []
                for label, score in zip(classification["labels"], classification["scores"]):
                    if score > self.min_prob:
                        topic_description.append((label, score))

            # Single label assignment
            elif classification["scores"][0] > self.min_prob:
                topic_description = [(classification["labels"][0], classification["scores"][0])]

            # Make sure that 10 items are returned
            if len(topic_description) == 0:
                topic_description = topics[topic]
            elif len(topic_description) < 10:
                topic_description += [("", 0) for _ in range(10 - len(topic_description))]
            updated_topics[topic] = topic_description

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = ", ".join(list(zip(*topics[topic]))[0])

        prompt = self.prompt
        if "[KEYWORDS]" in prompt:
            prompt = prompt.replace("[KEYWORDS]", keywords)
        if "[DOCUMENTS]" in prompt:
            to_replace = ""
            for doc in docs:
                to_replace += f"- {doc}\n"
            prompt = prompt.replace("[DOCUMENTS]", to_replace)

        return prompt
