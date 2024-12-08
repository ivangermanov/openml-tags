import json
import os
import time
from typing import List, Callable, Any

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

def replace_documents(prompt: str, docs: List[str]) -> str:
    to_replace = ""
    for doc in docs:
        to_replace += f"- {doc}\n---\n"
    return prompt.replace("[DOCUMENTS]", to_replace)

def create_prompt(prompt: str, docs: List[str], topic: Any, topics: dict) -> str:
    keywords = list(zip(*topics[topic]))[0]

    if prompt == DEFAULT_CHAT_PROMPT or prompt == DEFAULT_PROMPT:
        prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
        prompt = replace_documents(prompt, docs)
    else:
        if "[KEYWORDS]" in prompt:
            prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
        if "[DOCUMENTS]" in prompt:
            prompt = replace_documents(prompt, docs)

    return prompt

def fetch_response_with_retry(
    fetch_func: Callable,
    max_retries: int = 3,
    delay: float = 2,
    *args,
    **kwargs
) -> Any:
    retries = 0
    while retries < max_retries:
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
                time.sleep(delay)
            else:
                raise e

def append_to_json(file_path, new_data):
    """Append new data to a JSON array stored in a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
            # Load existing data
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
            # Append new data
            data.append(new_data)
            # Move the file pointer to the beginning
            file.seek(0)
            # Write the updated data
            json.dump(data, file, indent=2)
            # Truncate the file to the current size
            file.truncate()
    else:
        with open(file_path, 'w') as file:
            # Write new data in a list
            json.dump([new_data], file, indent=2)