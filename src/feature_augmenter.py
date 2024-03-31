import pandas as pd


class FeatureAugmenter:
    def __init__(self, description_column: str, features_column: str, augmented_column: str) -> None:
        """
        Initializes the FeatureAugmenter with the names of the feature, description, and augmented columns.

        :param features_column: The name of the column containing the features.
        :param description_column: The name of the column containing the description.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        """
        self.description_column = description_column
        self.features_column = features_column
        self.augmented_column = augmented_column

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by updating a custom column with the feature information appended to the existing or original description.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        features = df[self.features_column]
        augmented_description = df.get(self.augmented_column, df[self.description_column])

        def join_features(feature_list):
            if isinstance(feature_list, dict):
                return ', '.join(str(feature.name) for feature in feature_list.values())
            return ''

        features_str = features.apply(join_features)
        augmented_description = 'Features: ' + features_str + '\n\n' + augmented_description
        df[self.augmented_column] = augmented_description

        return df
