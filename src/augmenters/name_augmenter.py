import pandas as pd


class NameAugmenter:
    """
    A class to create or update a custom column in a DataFrame by prepending the .name attribute to the description.
    """

    def __init__(self, description_column: str, name_column: str, augmented_column: str) -> None:
        """
        Initializes the NameAugmenter with the names of the description and name columns, and the name of the
        augmented column.

        :param description_column: The name of the column containing the description.
        :param name_column: The name of the column containing the objects with a .name attribute.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        """
        self.description_column = description_column
        self.name_column = name_column
        self.augmented_column = augmented_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by updating a custom column with the .name attribute prepended to the existing or
        original description.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        names = df[self.name_column]
        augmented_description = df.get(self.augmented_column, df[self.description_column])

        # print(names_str)
        augmented_description = 'Name: ' + names + '\n\n' + augmented_description
        df[self.augmented_column] = augmented_description

        return df
