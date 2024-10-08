import time
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any
from bertopic.representation._base import BaseRepresentation
from mistralai import Mistral
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json
import os

from ._utils import (
    DEFAULT_PROMPT,
    DEFAULT_CHAT_PROMPT,
    create_prompt,
    fetch_response_with_retry,
    append_to_json
)


class MistralAI(BaseRepresentation):
    def __init__(
        self,
        client: Mistral,
        model: str = "mistral-large-latest",
        prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        nr_docs: int = 4,
        max_tokens: int = 2048,
        cache: bool = True,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_CHAT_PROMPT
        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.nr_docs = nr_docs
        self.max_tokens = max_tokens
        self.cache = cache
        self.prompts_ = []
        self.tokenizer = MistralTokenizer.from_model(model)
        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = self.generator_kwargs.pop("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        print("Extracting")
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs
        )

        updated_topics = {}
        for topic, docs in tqdm(
            repr_docs_mappings.items(), disable=not topic_model.verbose
        ):
            truncated_docs = []
            current_tokens = 0
            for doc in docs:
                # doc_tokens = self.tokenizer(doc)
                doc_tokens = len(self.tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[{"role": "user", "content": [{"type": "text", "text": doc}]}])).tokens)
                if current_tokens + doc_tokens <= self.max_tokens:
                    truncated_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    break

            prompt = create_prompt(self.prompt, truncated_docs, topic, topics)
            # print(prompt)
            self.prompts_.append(prompt)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            try:
                output_text = fetch_response_with_retry(
                    self.client.chat.complete,
                    max_retries=10,
                    delay=self.delay_in_seconds or 2,
                    model=self.model,
                    messages=messages,
                    **self.generator_kwargs,
                )
                print(output_text.choices[0].message.content)
                label = output_text.choices[0].message.content
                if self.cache:
                    os.makedirs("./extensions/_cache", exist_ok=True)
                    json_label = json.loads(label)
                    append_to_json("./extensions/_cache/mistral.json", json_label)
                updated_topics[topic] = [(label, 1)]
            except Exception as e:
                print(f"Error occurred: {e}")
                updated_topics[topic] = [("Error", 1)]

        return updated_topics
