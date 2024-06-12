import time
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document
from together import Together
from requests.exceptions import ChunkedEncodingError

DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_CHAT_PROMPT = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, can you give a short label of the topic?
"""


def fetch_response_with_retry(
    client, model, messages, generator_kwargs, max_retries=3, delay=2
):
    retries = 0
    while retries < max_retries:
        try:
            kwargs = {"model": model, "messages": messages, **generator_kwargs}
            stream = client.chat.completions.create(stream=True, **kwargs)
            output_text = ""
            for chunk in stream:
                output_text += chunk.choices[0].delta.content or ""
            return output_text
        except ChunkedEncodingError:
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(delay)
            else:
                raise


class TogetherAI(BaseRepresentation):
    """Using the TogetherAI API to generate topic labels based
    on one of their chat models.

    Arguments:
        client: A `together.Together` client
        model: Model to use within TogetherAI, defaults to `"meta-llama/Llama-3-70b-chat-hf"`.
        generator_kwargs: Kwargs passed to `together.Together.chat.completions.create`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        nr_docs: The number of documents to pass to TogetherAI if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to TogetherAI.
                   Accepts values between 0 and 1. A higher
                   values results in passing more diverse documents
                   whereas lower values passes more similar documents.
        doc_length: The maximum length of each document. If a document is longer,
                    it will be truncated. If None, the entire document is passed.
        tokenizer: The tokenizer used to calculate to split the document into segments
                   used to count the length of a document.
                       * If tokenizer is 'char', then the document is split up
                         into characters which are counted to adhere to `doc_length`
                       * If tokenizer is 'whitespace', the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and trunctated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`
        max_length: The maximum number of tokens in the prompt, including documents and keywords

    Usage:

    To use this, you will need to install the together package first:

    `pip install together`

    Then, get yourself an API key and use TogetherAI's API as follows:

    ```python
    from together import Together
    from bertopic.representation import TogetherAI
    from bertopic import BERTopic

    # Create your representation model
    client = Together(api_key='your_api_key')
    representation_model = TogetherAI(client, delay_in_seconds=5)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = TogetherAI(client, prompt=prompt, delay_in_seconds=5)
    ```
    """

    def __init__(
        self,
        client,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        max_length: int = 2048,
    ):
        self.client = client
        self.model = model

        if prompt is None:
            self.prompt = DEFAULT_CHAT_PROMPT
        else:
            self.prompt = prompt

        self.default_prompt_ = DEFAULT_CHAT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts_ = []

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
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
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Generate using TogetherAI's Language Model
        updated_topics = {}
        for topic, docs in tqdm(
            repr_docs_mappings.items(), disable=not topic_model.verbose
        ):
            truncated_docs = []
            current_length = 0
            for doc in docs:
                truncated_doc = truncate_document(
                    topic_model, self.doc_length, self.tokenizer, doc
                )
                doc_length = len(self.tokenizer.encode(truncated_doc))
                if current_length + doc_length <= self.max_length:
                    truncated_docs.append(truncated_doc)
                    current_length += doc_length
                else:
                    break

            prompt = self._create_prompt(truncated_docs, topic, topics)
            print(prompt)
            self.prompts_.append(prompt)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            try:
                max_retries = 3
                output_text = fetch_response_with_retry(self.client, self.model, messages, self.generator_kwargs, max_retries=max_retries)
                print(output_text)
                label = output_text.strip().replace("topic: ", "")
                updated_topics[topic] = [(label, 1)]
            except ChunkedEncodingError:
                updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT or self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n---\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt
