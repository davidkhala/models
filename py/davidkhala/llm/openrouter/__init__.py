from openrouter import OpenRouter
from openrouter.components import Model
from openrouter.errors import UnauthorizedResponseError
from openrouter.operations import ListData

from davidkhala.llm.model import Connectable, SDKProtocol
from davidkhala.llm.model.chat import CompareChatAware, on_response
from davidkhala.llm.model.garden import GardenAlike


class Client(CompareChatAware, Connectable, SDKProtocol, GardenAlike):
    def __init__(self, api_key: str):
        super().__init__()
        self.client: OpenRouter = OpenRouter(api_key)

    def chat(self, *user_prompt) -> str:
        """
        openrouter has no `n` parameter support
        """
        r = self.client.chat.send(
            model=self.model,
            models=self._models,
            messages=self.messages_from(*user_prompt)
        )
        return on_response(r, None)[0]

    def connect(self):
        try:
            _ = self.models
            return True
        except UnauthorizedResponseError:
            return False

    def list_models(self) -> list[Model]:
        return self.client.models.list().data


class Admin:
    def __init__(self, provisioning_key: str):
        self.provisioning_key = provisioning_key
        self.client = OpenRouter(provisioning_key)

    @property
    def keys(self) -> list[ListData]:
        return self.client.api_keys.list().data
