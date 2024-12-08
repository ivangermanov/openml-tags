import pandas as pd

data_folder = '../notebooks/data'


class LLMPromptAugmenter:
    def __init__(self, prompt_description_column: str):
        """
        Initializes the LLMPromptAugmenter with the name of the prompt description column.

        :param prompt_description_column: The name of the column to put the prompt description.
        """
        self.prompt_description_column = prompt_description_column

    def __call__(self, df: pd.DataFrame):
        """
        Augments a DataFrame by updating a custom column with the prompt description in a new column 'prompt_description_column'.
        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        df[self.prompt_description_column] = ''

        indices_to_remove = []

        for index in range(len(df.index)):
            try:
                with open(f"{data_folder}/dataset_{df.iloc[index]['dataset_id']}.txt", 'r') as f:
                    df.loc[df['dataset_id'] == df.iloc[index]['dataset_id'], self.prompt_description_column] = f.read()
            except FileNotFoundError:
                print(f"FileNotFoundError: dataset_{df.iloc[index]['dataset_id']}.txt not found")
                print(f"Removing row {index} from the DataFrame.")
                indices_to_remove.append(index)

        df.drop(df.index[indices_to_remove], inplace=True)

        return df
