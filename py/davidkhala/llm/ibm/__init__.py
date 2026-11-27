from typing import Literal

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference, Embeddings
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from pydantic import BaseModel

from davidkhala.llm.model.chat import ChoicesChat, DeterministicChat, ChoicesAware
from davidkhala.llm.model.garden import GardenAlike
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.prompt.param import Prompt


class ID(BaseModel):
    id: str
    type: Literal["Chat", "Embedding", "TimeSeries", "AudioTranscriptions"]


class Client(ChoicesChat, DeterministicChat, GardenAlike, EmbeddingAware):
    def __init__(self, project_id, *, region: str, api_key: str):
        super().__init__()
        self.client = APIClient(Credentials(
            url=f"https://{region}.ml.cloud.ibm.com",
            api_key=api_key
        ), project_id)
        self.handler: ModelInference | Embeddings

    def as_embeddings(self, model: str):
        self.handler = Embeddings(
            model_id=model,
            api_client=self.client,
        )

    def encode(self, *texts: str) -> list[list[float]]:
        return self.handler.embed_documents(texts=list(texts))

    def list_models(self) -> list[ID]:
        return [
            ID(id=m, type="Chat") for m in self.client.foundation_models.ChatModels
        ] + [
            ID(id=m, type="Embedding") for m in
            self.client.foundation_models.EmbeddingModels
        ] + [
            ID(id=m, type="TimeSeries") for m in self.client.foundation_models.TimeSeriesModels
        ] + [
            ID(id=m, type="AudioTranscriptions") for m in self.client.foundation_models.AudioTranscriptionsModels
        ]

    def as_chat(self, model: str | None, sys_prompt: str = None):
        super().as_chat(model=None, sys_prompt=sys_prompt)
        self.handler = ModelInference(
            model_id=model,
            api_client=self.client,
        )

    def chat(self, *user_prompt: Prompt):
        response: dict = self.handler.chat(
            messages=self.messages_from(*user_prompt),
            params=TextChatParameters(
                seed=self.seed,
                n=self.n
            )
        )
        return ChoicesAware.model_validate(response)
