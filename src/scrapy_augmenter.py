import pandas as pd

data_folder = '../scrapy/openml_tags/openml_tags/spiders/data'
archive_ics_uci_path = f'{data_folder}/archive_ics_uci_edu.json'
ebi_ac_uk_path = f'{data_folder}/ebi_ac_uk.json'


class ScrapyAugmenter:
    def __init__(self, description_column: str, scraped_column: str, augmented_column: str):
        """
        Initializes the ScrapyAugmenter with the names of the description, scraped, and augmented columns.

        :param description_column: The name of the column containing the description.
        :param scraped_column: The name of the column where the scraping data will be stored.
        :param augmented_column: The name of the column to be created or modified with the augmented description.
        """
        self.description_column = description_column
        self.scraped_column = scraped_column
        self.augmented_column = augmented_column

    def __call__(self, df: pd.DataFrame):
        """
        Augments a DataFrame by updating a custom column with the scraped information appended to the existing or original description.

        :param df: A pandas DataFrame.
        :return: The augmented pandas DataFrame with an updated custom column.
        """
        archive_ics_uci_df = pd.read_json(archive_ics_uci_path)
        ebi_ac_uk_df = pd.read_json(ebi_ac_uk_path)

        df[self.scraped_column] = ''
        for index, row in archive_ics_uci_df.iterrows():
            df.loc[df['dataset_id'] == row['dataset_id'], self.scraped_column] = row['scraped_data']
        for index, row in ebi_ac_uk_df.iterrows():
            df.loc[df['dataset_id'] == row['dataset_id'], self.scraped_column] = row['scraped_data']

        augmented_description = df.get(self.augmented_column, df[self.description_column])

        if not df[self.scraped_column].empty:
            df[self.augmented_column] = 'Scraped Data: ' + df[self.scraped_column] + '\n\n' + augmented_description

        return df
