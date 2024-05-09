from typing import List, Callable

import pandas as pd


class DatasetAugmenter:
    """
    A class to apply a series of augmentations to a pandas DataFrame.
    """

    def __init__(self, augmenters: List[Callable[[pd.DataFrame], pd.DataFrame]]) -> None:
        """
        Initializes the DatasetAugmenter with a list of augmentation callables.

        :param augmenters: A list of callables that take a DataFrame as input and return an augmented DataFrame.
        """
        self.augmenters = augmenters

    def _apply_augmentations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all registered augmentations to a DataFrame.

        :param df: A pandas DataFrame to be augmented.
        :return: A new pandas DataFrame containing the augmented rows.
        """
        for augmenter in self.augmenters:
            df = augmenter(df)
        return df

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all augmentations to a DataFrame.

        :param df: A pandas DataFrame to be augmented.
        :return: A new pandas DataFrame containing the augmented rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input must be a pandas DataFrame.")

        augmented_df = self._apply_augmentations(df)
        return augmented_df
