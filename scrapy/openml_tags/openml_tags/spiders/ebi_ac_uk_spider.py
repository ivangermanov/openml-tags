from scrapy.http import Response

from .base_spider import BaseSpider

dataset_info_xpath = ("//div[contains(text(), 'Dataset Information') or contains(., 'Dataset "
                      "Information')]/ancestor::div[contains(@class, 'shadow')]//text()")


class EbiAcUkSpider(BaseSpider):
    name = "ebi_ac_uk"

    @property
    def allowed_domains(self):
        return ["ebi.ac.uk"]

    async def parse(self, response: Response, **kwargs):
        page = response.meta['playwright_page']
        # content = await page.content()

        dataset_info_texts = response.xpath(dataset_info_xpath).getall()
        dataset_info = ' '.join([text.strip() for text in dataset_info_texts if text.strip()])

        await page.close()

        return {'dataset_id': response.meta['dataset_id'], 'url': response.url, 'dataset_info': dataset_info}
