from typing import Iterator

from openrouter import OpenRouter
from openrouter.operations import ListData, ListResponse

from davidkhala.llm.openrouter import yield_list


class Admin:
    def __init__(self, provisioning_key: str):
        super().__init__()
        self.provisioning_key = provisioning_key
        self.client = OpenRouter(provisioning_key)

    def list_keys(self) -> Iterator[ListData]:
        page: ListResponse[ListData] | None = self.client.api_keys.list()
        return yield_list(page)

    @property
    def keys(self):
        return [_.name for _ in self.list_keys()]
