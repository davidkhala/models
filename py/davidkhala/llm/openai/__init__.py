from typing import Literal

from openai import OpenAI, AuthenticationError, PermissionDeniedError
from openai.types import Model

from davidkhala.llm.model import Connectable
from davidkhala.llm.model.chat import on_response, ChoicesChat
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.garden import GardenAlike


class Client(ChoicesChat, EmbeddingAware, GardenAlike, Connectable):
    @property
    def free_models(self) -> list[str]:
        return [] # no free model

    def __init__(self, client: OpenAI):
        super().__init__()
        self.client: OpenAI = client
        self.encoding_format: Literal["float", "base64"] = "float"

    def connect(self):
        try:
            self.list_models()
            return True
        except AuthenticationError | PermissionDeniedError:
            return False

    def list_models(self) -> list[Model]:
        return self.client.models.list().data

    def encode(self, *_input: str) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(_input),
            encoding_format=self.encoding_format
        )
        return [item.embedding for item in response.data]

    def chat(self, *user_prompt, **kwargs):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages_from(*user_prompt),
            n=self.n,
            **kwargs
        )

        return on_response(response, self.n)

    def close(self):
        self.client.close()
