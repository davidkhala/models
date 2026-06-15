from typing import Sequence

from ollama import Client as RawClient
from ollama._types import WebSearchResult, WebFetchResponse


class Client:
    def __init__(self, api_key: str):
        self.client = RawClient(
            headers={
                'authorization': f'Bearer {api_key}'
            }
        )

    def search(self, message: str) -> Sequence[WebSearchResult]:
        return self.client.web_search(message).results

    def fetch(self, url: str) -> WebFetchResponse:
        return self.client.web_fetch(url)
