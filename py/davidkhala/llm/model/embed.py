from abc import ABC, abstractmethod

from davidkhala.llm.model import ModelAware


class EmbeddingAware(ModelAware, ABC):
    def as_embeddings(self, model: str):
        self.model = model

    @abstractmethod
    def encode(self, *_input: str) -> list[list[float]]: ...
