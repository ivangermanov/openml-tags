import pandas as pd


class FeatureAugmenter:
    def __init__(self, description_column: str, features_column: str, augmented_column: str, max_features: int = 10000,
                 reduce_features: bool = False) -> None:
        """
        Initializes the FeatureAugmenter with the names of the feature, description, and augmented columns, the maximum number of features to include, and a flag to indicate if the features should be reduced.

        :param features_column: The name of the column containing the features.
        :param description_column: The name of the column containing the description.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        :param max_features: The maximum number of features to include in the augmented description. Default is 10000.
        :param reduce_features: Boolean flag indicating if the features column should be truncated to the max_features. Default is False.
        """
        self.description_column = description_column
        self.features_column = features_column
        self.augmented_column = augmented_column
        self.max_features = max_features
        self.reduce_features = reduce_features

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments a DataFrame by updating a custom column with the feature information appended to the existing or original description.
        Optionally, truncates the features in the features column if reduce_features is True.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        features = df[self.features_column]

        # Optionally truncate the features in the features column
        if self.reduce_features:
            df[self.features_column] = df[self.features_column].apply(
                lambda x: dict(list(x.items())[:self.max_features]) if isinstance(x, dict) else x)

        augmented_description = df.get(self.augmented_column, df[self.description_column])

        def join_features(feature_list):
            if isinstance(feature_list, dict):
                feature_names = [str(feature.name) for feature in feature_list.values()]
                if len(feature_names) > self.max_features:
                    feature_names = feature_names[:self.max_features]
                    feature_names.append(f"and {len(feature_list) - self.max_features} more...")
                return ', '.join(feature_names)
            return ''

        features_str = features.apply(join_features)
        augmented_description = augmented_description + '\n\n' + 'Features: ' + features_str
        df[self.augmented_column] = augmented_description

        return df
