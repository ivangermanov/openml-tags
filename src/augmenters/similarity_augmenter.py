import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAugmenter:
    """
    A class to filter out similar rows from a DataFrame based on a similarity threshold.
    """

    def __init__(self, description_column: str, augmented_column: str, similar_datasets_column: str,
                 threshold: float = 0.9999999):
        """
        Initialize the SimilarityAugmenter
        """
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.description_column = description_column
        self.augmented_column = augmented_column
        self.similar_datasets_column = similar_datasets_column
        self.threshold = threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by filtering out similar rows based on a similarity threshold.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with similar rows removed and their dataset_ids attached to the remaining row.
        """
        descriptions = df.get(self.augmented_column, df[self.description_column])

        # Compute the TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(descriptions)

        # Compute the cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create a mask to ignore the diagonal (self-similarity)
        np.fill_diagonal(similarity_matrix, 0)

        # Find the indices of similar rows
        similar_indices = np.where(similarity_matrix > self.threshold)

        # Create a dictionary to store the indices of rows to drop and their corresponding dataset_ids
        drop_indices = {}

        # Iterate over the similar row indices
        for i, j in zip(*similar_indices):
            if i < j and i not in drop_indices:
                drop_indices[j] = df.iloc[j]['dataset_id']

        # Create a dictionary to store the dataset_ids of similar rows for each remaining row
        similar_datasets = {}

        # Iterate over the remaining rows
        for i in range(len(df)):
            if i not in drop_indices:
                similar_datasets[i] = [drop_indices[j] for j in drop_indices if
                                       similarity_matrix[i, j] > self.threshold]

        # Drop the similar rows
        df = df.drop(df.index[list(drop_indices.keys())])
        # Attach the similar dataset_ids to the remaining rows
        df[self.similar_datasets_column] = df.index.map(similar_datasets)

        return df
