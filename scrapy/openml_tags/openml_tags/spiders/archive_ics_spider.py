from scrapy.http import Response

from .base_spider import BaseSpider

dataset_info_xpath = "//div[contains(text(), 'Dataset Information') or contains(., 'Dataset Information')]/ancestor::div[contains(@class, 'shadow')]//text()"

variables_table_xpath = "//div[contains(text(), 'Variables Table') or contains(., 'Variables Table')]/ancestor::div[contains(@class, 'shadow')]//text()"

additional_info_xpath = "//div[contains(text(), 'Additional Variable Information') or contains(., 'Additional Variable Information')]/ancestor::div[contains(@class, 'shadow')]//text()"


class ArchiveIcsSpider(BaseSpider):
    name = "archive_ics_uci_edu"

    @property
    def allowed_domains(self):
        return ["archive.ics.uci.edu"]

    async def parse(self, response: Response, **kwargs):
        dataset_info_texts = response.xpath(dataset_info_xpath).getall()
        dataset_info = ' '.join([text.strip() for text in dataset_info_texts if text.strip()])

        additional_info_texts = response.xpath(additional_info_xpath).getall()
        additional_info = ' '.join([text.strip() for text in additional_info_texts if text.strip()])

        page = response.meta['playwright_page']
        select_primary = await page.query_selector('.select-primary')

        if select_primary:
            await page.click('.select-primary')
            await page.select_option('.select-primary', '25')
            next_page_button_selector = 'button[aria-label="Next Page"]'
            # Loop until the "Next Page" button is disabled
            is_next_page = True
            # variables_table = ''
            max_clicks = 4
            click_count = 0
            while is_next_page and click_count < max_clicks:
                # Extract the variables table only
                # content = await page.content()
                # sel = scrapy.Selector(text=content)

                # variables_table_texts = sel.xpath(variables_table_xpath).getall()
                # variables_table = ' '.join([text.strip() for text in variables_table_texts if text.strip()])

                # Check if the "Next Page" button is disabled, i.e. check next_page_button_selector if it has the
                # disabled attribute
                next_button = await page.query_selector(next_page_button_selector)
                if next_button:
                    is_disabled = await page.evaluate('(element) => element.disabled', next_button)
                    if is_disabled:
                        is_next_page = False
                    else:
                        await page.click(next_page_button_selector)
                        click_count += 1
                else:
                    is_next_page = False

        await page.close()

        return {'dataset_id': response.meta['dataset_id'], 'url': response.url, 'dataset_info': dataset_info,
                'additional_info': additional_info}  # 'variables_table': variables_table}
