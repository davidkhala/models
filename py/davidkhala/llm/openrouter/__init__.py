from openrouter import OpenRouter
from openrouter.components import Model
from openrouter.errors import UnauthorizedResponseError
from openrouter.operations import ListData, CreateEmbeddingsResponseBody
from typing_extensions import Literal

from davidkhala.llm.model import Connectable
from davidkhala.llm.model.chat import on_response, Prompt
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.openrouter import OpenRouterModel


class Client(OpenRouterModel, EmbeddingAware, Connectable):
    @property
    def free_models(self) -> list[str]:
        return ['openrouter/free', *(_.id for _ in self.list_models() if _.id.endswith(':free'))]

    def __init__(self, api_key: str):
        super().__init__()
        self.client: OpenRouter = OpenRouter(api_key)

    def chat(self, *user_prompt: Prompt) -> str:
        """
        python SDK Do not support FilePrompt yet
        """
        plugins = []
        self.messages_from(*user_prompt)
        r = self.client.chat.send(
            model=self.model,
            models=self._models,
            messages=self.messages,
            plugins=plugins
        )
        return on_response(r, OpenRouterModel.n)[0]

    def connect(self):
        try:
            _ = self.models
            return True
        except UnauthorizedResponseError:
            return False

    def list_models(self, _type: Literal['embeddings'] | None = None) -> list[Model]:
        match _type:
            case 'embeddings':
                return self.client.embeddings.list_models().data
        return self.client.models.list().data

    def encode(self, *_input: str) -> list[list[float]]:
        r: CreateEmbeddingsResponseBody = self.client.embeddings.generate(
            input=_input,
            model=self.model
        )
        return [_.embedding for _ in r.data]


class Admin:
    def __init__(self, provisioning_key: str):
        super().__init__()
        self.provisioning_key = provisioning_key
        self.client = OpenRouter(provisioning_key)

    @property
    def keys(self) -> list[ListData]:
        return self.client.api_keys.list().data
