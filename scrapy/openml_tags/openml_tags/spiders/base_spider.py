import json
import os
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path
from typing import Iterable

import scrapy
from scrapy import Request
from scrapy.http import Response

data_dir = 'data'


class BaseSpider(scrapy.Spider, ABC):
    custom_settings = {'FEED_FORMAT': 'json', 'DOWNLOAD_DELAY': 5.0, 'CONCURRENT_REQUESTS_PER_DOMAIN': 1, }

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        settings.set('FEED_URI', os.path.join(data_dir, cls.data_file()))

    @property
    @abstractmethod
    def allowed_domains(self) -> Iterable[str]:
        """
        Subclasses should override this to return the allowed domains.
        """
        raise NotImplementedError("Subclasses should override property 'allowed_domains'")

    @property
    def datasets_file(self) -> str:
        """
        File where the dataset URLs are saved.
        """
        return f'openml_datasets_{self.name}.json'

    @classmethod
    def data_file(cls):
        """
        File where the results are saved.
        """
        return f'{cls.name}_result.json'

    @property
    def error_file(self) -> str:
        """
        File where the errors are logged.
        """
        return f'{self.name}_errors.log'

    def start_requests(self) -> Iterable[Request]:
        json_file_path = Path(__file__).parent.parent.parent.parent / 'datasets' / self.datasets_file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            self.log(f"Reading file {f.name}: {len(data)}")

        for i, entry in enumerate(data):
            url = entry['original_data_url']
            self.log(f"Processing {i}: {url}")
            if any(domain in url for domain in self.allowed_domains):
                yield Request(url=url, callback=self.parse, errback=self.errback,
                              meta={'dataset_id': entry['dataset_id'], 'playwright': True,
                                    'playwright_include_page': True})

    @abstractmethod
    async def parse(self, response: Response, **kwargs):
        # Override this method in the specific spider classes
        raise NotImplementedError("Subclasses should override method 'parse'")

    async def errback(self, failure):
        self.logger.error("Error here: ", repr(failure))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(data_dir, self.error_file), 'a') as f:
            f.write(f"{timestamp}: {failure.request.url}\n")

        page = failure.request.meta['playwright_page']
        await page.close()