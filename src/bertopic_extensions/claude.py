import time
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any
from bertopic.representation._base import BaseRepresentation
from anthropic import Anthropic
from anthropic.types import ContentBlock

from ._utils import (
    DEFAULT_PROMPT,
    DEFAULT_CHAT_PROMPT,
    create_prompt,
    fetch_response_with_retry,
)


class ClaudeAI(BaseRepresentation):
    """Using the Claude API to generate topic labels based on Claude 3.5.

    Arguments:
        client: An `anthropic.Anthropic` client
        model: Model to use within Claude, defaults to `"claude-3-sonnet-20240229"`.
        generator_kwargs: Kwargs passed to `anthropic.Anthropic.messages.create`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        nr_docs: The number of documents to pass to Claude if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        max_tokens: The maximum number of tokens in the prompt, including documents and keywords

    Usage:

    To use this, you will need to install the anthropic package first:

    `pip install anthropic`

    Then, get yourself an API key and use Claude's API as follows:

    ```python
    from anthropic import Anthropic
    from bertopic.representation import ClaudeAI
    from bertopic import BERTopic

    # Create your representation model
    client = Anthropic(api_key='your_api_key')
    representation_model = ClaudeAI(client, delay_in_seconds=5)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = ClaudeAI(client, prompt=prompt, delay_in_seconds=5)
    ```
    """

    def __init__(
        self,
        client: Anthropic,
        model: str = "claude-3-sonnet-20240229",
        prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        nr_docs: int = 4,
        max_tokens: int = 2048,
    ):
        self.client = client
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_CHAT_PROMPT
        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.nr_docs = nr_docs
        self.max_tokens = max_tokens
        self.prompts_ = []

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

        # Generate using Claude's Language Model
        updated_topics = {}
        for topic, docs in tqdm(
            repr_docs_mappings.items(), disable=not topic_model.verbose
        ):
            truncated_docs = []
            current_tokens = 0
            for doc in docs:
                doc_tokens = self.client.count_tokens(doc)
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
                    self.client.messages.create,
                    max_retries=3,
                    delay=self.delay_in_seconds or 2,
                    model=self.model,
                    messages=messages,
                    **self.generator_kwargs,
                )
                print(output_text.content[0].text.strip())
                label = output_text.content[0].text.strip()
                updated_topics[topic] = [(label, 1)]
            except Exception as e:
                print(f"Error occurred: {e}")
                updated_topics[topic] = [("Error", 1)]

        return updated_topics
