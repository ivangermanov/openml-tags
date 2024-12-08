import pandas as pd

class IdAugmenter:
    """
    A class to create or update a custom column in a DataFrame by prepending the ID to the description.
    """

    def __init__(self, description_column: str, id_column: str, augmented_column: str) -> None:
        """
        Initializes the IdAugmenter with the names of the description, ID, and augmented columns.

        :param description_column: The name of the column containing the description.
        :param id_column: The name of the column containing the IDs.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        """
        self.description_column = description_column
        self.id_column = id_column
        self.augmented_column = augmented_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by updating a custom column with the ID information prepended to the existing or original description.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        augmented_description = df.get(self.augmented_column, df[self.description_column])
        df[self.augmented_column] = 'ID: ' + df[self.id_column].astype(str) + '\n\n' + augmented_description

        return df