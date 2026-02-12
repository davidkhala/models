from davidkhala.llm.model import ModelAware


class EmbeddingAware(ModelAware):
    def as_embeddings(self, model: str):
        self.model = model

    def encode(self, *_input: str) -> list[list[float]]: ...
