from typing import Literal

from openrouter import OpenRouter
from openrouter.components import ChatAssistantMessageContent, ChatResult, Model
from openrouter.errors import UnauthorizedResponseError
from openrouter.operations import CreateEmbeddingsResponseBody

from davidkhala.llm.model import Connectable
from davidkhala.llm.model.chat import DeterministicChat
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.garden import GardenAlike, TrialAvailable
from davidkhala.llm.model.openrouter import OpenRouterModel, Plugins
from davidkhala.llm.model.prompt.param import Prompt
from davidkhala.llm.openrouter import yield_list


class Client(OpenRouterModel, GardenAlike, DeterministicChat, EmbeddingAware, Connectable, TrialAvailable):
    @property
    def free_models(self) -> list[str]:
        return ['openrouter/free', *(_.id for _ in self.list_models() if _.id.endswith(':free'))]

    def __init__(self, api_key: str):
        super().__init__()
        self.client: OpenRouter = OpenRouter(api_key)

    def chat(self, *user_prompt: Prompt,
             pdf_engine: Plugins.PDF_ENGINE = 'cloudflare-ai') -> ChatAssistantMessageContent:
        plugins = [
            Plugins.pdf(pdf_engine)
        ]

        r: ChatResult = self.client.chat.send(
            model=self.model,
            models=self._models,
            messages=self.messages_from(*user_prompt),
            plugins=plugins,
            seed=self.seed,
        )

        return r.choices[0].message.content

    def connect(self):
        try:
            _ = self.models
            return True
        except UnauthorizedResponseError:
            return False

    def list_models(self, _type: Literal['embeddings'] | None = None) -> list[Model]:
        match _type:
            case 'embeddings':
                from openrouter.operations import ListEmbeddingsModelsResponse
                _: ListEmbeddingsModelsResponse[Model] | None = self.client.embeddings.list_models()
                return list(yield_list(_))
            case _:
                from openrouter.operations import GetModelsResponse
                _: GetModelsResponse[Model] | None = self.client.models.list()
                return list(yield_list(_))

    def encode(self, *_input: str) -> list[list[float]]:
        r: CreateEmbeddingsResponseBody = self.client.embeddings.generate(
            input=_input,
            model=self.model
        )
        return [_.embedding for _ in r.data]
