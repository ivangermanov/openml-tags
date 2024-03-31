import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAugmenter:
    """
    A class to filter out similar rows from a DataFrame based on a similarity threshold.
    """

    def __init__(self, description_column: str, augmented_column: str, threshold: float = 0.99):
        """
        Initialize the SimilarityAugmenter
        """
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.description_column = description_column
        self.augmented_column = augmented_column
        self.threshold = threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by filtering out similar rows based on a similarity threshold.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with similar rows removed.
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

        # Create a set to store the indices of rows to drop
        drop_indices = set()

        # Iterate over the similar row indices
        for i, j in zip(*similar_indices):
            if i < j and i not in drop_indices:
                drop_indices.add(j)

        # Drop the similar rows
        df = df.drop(df.index[list(drop_indices)])

        return df
