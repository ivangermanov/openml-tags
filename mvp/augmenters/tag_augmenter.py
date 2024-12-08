import json
import pandas as pd
from typing import Optional

class TagAugmenter:
    """
    A class to create or update a custom column in a DataFrame by prepending the tags to the description,
    with automatic tag removal if a JSON file is provided.
    """

    def __init__(self, description_column: str, tag_column: str, augmented_column: str, dataset_id_column: str, json_file_path: Optional[str] = None) -> None:
        """
        Initializes the TagAugmenter with the names of the description, tags, and dataset ID columns,
        the name of the augmented column, and optionally the path to the JSON file.

        :param description_column: The name of the column containing the description.
        :param tag_column: The name of the column containing the tags.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        :param dataset_id_column: The name of the column containing the dataset IDs.
        :param json_file_path: The path to the JSON file containing tags to remove (optional).
        """
        self.description_column = description_column
        self.tag_column = tag_column
        self.augmented_column = augmented_column
        self.dataset_id_column = dataset_id_column
        self.json_file_path = json_file_path
        self.tags_to_remove = self._load_json_tags() if json_file_path else {}

    def _load_json_tags(self) -> dict:
        """
        Loads the JSON file containing tags to remove.

        :return: A dictionary with dataset IDs as keys and lists of tags to remove as values.
        """
        try:
            with open(self.json_file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            return {}

    def _process_tags(self, tags: list, dataset_id: str) -> str:
        """
        Processes tags, removing specified tags if a JSON file was provided.

        :param tags: List of tags for a dataset.
        :param dataset_id: The dataset ID.
        :return: Processed tags as a comma-separated string.
        """
        if isinstance(tags, list):
            if self.tags_to_remove and dataset_id in self.tags_to_remove:
                tags = [tag for tag in tags if tag not in self.tags_to_remove[dataset_id]]
            return ', '.join(tags)
        return ''

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by updating a custom column with the tag information prepended to the existing or original description.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        augmented_description = df.get(self.augmented_column, df[self.description_column])

        def process_row(row):
            return self._process_tags(row[self.tag_column], str(row[self.dataset_id_column]))

        tags_str = df.apply(process_row, axis=1)
        augmented_description = 'Tags: ' + tags_str + '\n\n' + augmented_description
        df[self.augmented_column] = augmented_description

        return df