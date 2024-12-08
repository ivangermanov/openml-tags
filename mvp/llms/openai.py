import csv
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any
from bertopic.representation._base import BaseRepresentation
import openai
import tiktoken

from ._utils import (
    DEFAULT_PROMPT,
    DEFAULT_CHAT_PROMPT,
    create_prompt,
    fetch_response_with_retry,
    append_to_json,
)


class OpenAIGPTRepresentation(BaseRepresentation):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = 5,
        nr_docs: int = 50,
        max_tokens_total: int = 128000,
        max_tokens_per_doc: int = 256,
        cache: bool = True,
    ):
        """
        Arguments:
            api_key: OpenAI API key
            model: Model to use within OpenAI, defaults to `"gpt-4o-mini"`.
            generator_kwargs: Kwargs passed to `openai.ChatCompletion.create`
                          for fine-tuning the output.
            prompt: The prompt to be used in the model.
            delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
            nr_docs: The number of documents to pass to GPT if a prompt
                     with the `["DOCUMENTS"]` tag is used.
            max_tokens_total: The maximum number of tokens in the entire prompt
            max_tokens_per_doc: The maximum number of tokens per document
            cache: Whether to cache the results
        """
        self.api_key = api_key
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_CHAT_PROMPT
        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.nr_docs = nr_docs
        self.max_tokens_total = max_tokens_total
        self.max_tokens_per_doc = max_tokens_per_doc
        self.cache = cache
        self.prompts_ = []

        # Set OpenAI API key
        openai.api_key = self.api_key

        # Initialize tokenizer for the given model
        self.tokenizer = tiktoken.encoding_for_model(self.model)

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = self.generator_kwargs.pop("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the local tokenizer."""
        return len(self.tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics with token limitations per document and total."""
        print("Extracting")
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs
        )

        # Generate using OpenAI's Language Model
        updated_topics = {}
        for topic, docs in tqdm(
            repr_docs_mappings.items(), disable=not topic_model.verbose
        ):
            print(f"Topic {topic} has {len(docs)} documents")
            # First truncate each document individually
            truncated_docs = [
                self.truncate_text(doc, self.max_tokens_per_doc)
                for doc in docs
            ]

            # Then ensure total tokens don't exceed max_tokens_total
            final_docs = []
            current_tokens = 0
            
            # Account for prompt tokens (approximate)
            base_prompt = create_prompt("", [], topic, topics)
            prompt_tokens = self.count_tokens(base_prompt)
            tokens_available = self.max_tokens_total - prompt_tokens

            for doc in truncated_docs:
                doc_tokens = self.count_tokens(doc)
                if current_tokens + doc_tokens <= tokens_available:
                    final_docs.append(doc)
                    current_tokens += doc_tokens
                else:
                    break

            prompt = create_prompt(self.prompt, final_docs, topic, topics)
            self.prompts_.append(prompt)

            messages = [
                {"role": "system", "content": "You are a highly intelligent text analysis assistant. Your task is to extract and synthesize themes, keywords, and overarching topics from the provided texts. Follow the instructions provided by the user carefully and ensure all responses are concise, structured, and in JSON format."},
                {"role": "user", "content": prompt},
            ]

            try:
                response = fetch_response_with_retry(
                    openai.chat.completions.create,
                    max_retries=10,
                    delay=self.delay_in_seconds,
                    model=self.model,
                    messages=messages,
                    **self.generator_kwargs,
                )
                # print(prompt)
                label = response.choices[0].message.content
                # print(label)
                if self.cache:
                    os.makedirs("./_cache", exist_ok=True)
                    json_label = json.loads(label)
                    # Save to individual topic file
                    with open(f"./_cache/{topic}.json", 'w', encoding='utf-8') as f:
                        json.dump(json_label, f, ensure_ascii=False, indent=4)
                    with open(f"./_cache/{topic}_prompt.txt", 'w', encoding='utf-8') as f:
                        f.write(prompt)
                updated_topics[topic] = [(label, 1)]
            except Exception as e:
                print(f"Error occurred: {e}")
                os.makedirs("./_cache", exist_ok=True)
                # write response variable to file
                with open(f"./_cache/{topic}_response.txt", 'w', encoding='utf-8') as f:
                    f.write(label)
                with open(f"./_cache/{topic}_prompt.txt", 'w', encoding='utf-8') as f:
                    f.write(prompt)
                # save error
                with open(f"./_cache/{topic}_error.txt", 'w', encoding='utf-8') as f:
                    f.write(str(e))
                updated_topics[topic] = [("Error", 1)]

        return updated_topics