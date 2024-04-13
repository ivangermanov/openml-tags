import scrapy
from bs4 import BeautifulSoup
from scrapy.http import Response

from .base_spider import BaseSpider


class EbiAcUkSpider(BaseSpider):
    name = "ebi_ac_uk"

    @property
    def allowed_domains(self):
        return ["ebi.ac.uk"]

    async def parse(self, response: Response, **kwargs):
        page = response.meta['playwright_page']
        await page.wait_for_timeout(10000)
        content = await page.content()
        sel = scrapy.Selector(text=content)
        iframes = sel.xpath('//div[contains(@class, "BCK-Details-Container")]//iframe').getall()
        scraped_data = ''

        for iframe in iframes:
            src = iframe.split('src="')[1].split('"')[0]
            self.log(f"Processing iframe: {src}")
            await page.goto(src)
            await page.wait_for_timeout(2000)
            content = await page.content()
            # print("CONTENT: ", BeautifulSoup(content, 'html.parser').get_text())
            # print("CONTENT: ", BeautifulSoup(content, 'html.parser').find('div', class_='main-container').get_text())
            scraped_data += BeautifulSoup(content, 'html.parser').get_text()
            scraped_data = scraped_data.replace('\nChEMBL - ChEMBL\n\n\n\n\n\n', '')
            await page.goto(response.url)

        await page.close()

        return {'dataset_id': response.meta['dataset_id'], 'url': response.url, 'scraped_data': scraped_data}
