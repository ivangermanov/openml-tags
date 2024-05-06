import pandas as pd

data_folder = '../notebooks/data'


class LLMPromptAugmenter:
    def __init__(self, prompt_description_column: str):
        """
        Initializes the ScrapyAugmenter with the names of the description, scraped, and augmented columns.

        :param prompt_description_column: The name of the column to put the prompt description.
        """
        self.prompt_description_column = prompt_description_column

    def __call__(self, df: pd.DataFrame):
        """
        Augments a DataFrame by updating a custom column with the prompt description in a new column 'prompt_description'.
        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        df[self.prompt_description_column] = ''

        # for index, row in df.iterrows():
        #     with open(f"{data_folder}/dataset_{index}.txt", 'r') as f:
        #         print(index)
        #         df.loc[index, self.prompt_description_column] = f.read()
        # do same but more efficiently without iterrows
        for index in range(len(df.index)):
            with open(f"{data_folder}/dataset_{index}.txt", 'r') as f:
                df.iloc[index, df.columns.get_loc(self.prompt_description_column)] = f.read()

        return df
