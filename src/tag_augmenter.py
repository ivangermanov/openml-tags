import pandas as pd


class TagAugmenter:
    """
    A class to create or update a custom column in a DataFrame by prepending the tags to the description.
    """

    def __init__(self, description_column: str, tag_column: str, augmented_column: str) -> None:
        """
        Initializes the TagAugmenter with the names of the description and tags columns, and the name of the augmented column.

        :param description_column: The name of the column containing the description.
        :param tag_column: The name of the column containing the tags.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        """
        self.description_column = description_column
        self.tag_column = tag_column
        self.augmented_column = augmented_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by updating a custom column with the tag information prepended to the existing or original description.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        tags = df[self.tag_column]
        augmented_description = df.get(self.augmented_column, df[self.description_column])

        def join_tags(tag_list):
            if isinstance(tag_list, list):
                return ', '.join(tag_list)
            return ''

        tags_str = tags.apply(join_tags)
        augmented_description = 'Tags: ' + tags_str + '\n\n' + augmented_description
        df[self.augmented_column] = augmented_description

        return df
