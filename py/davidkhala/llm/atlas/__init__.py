import voyageai

from davidkhala.llm.model.embed import EmbeddingAware


class Client(EmbeddingAware):
    def __init__(self, api_key):
        super().__init__()
        self.client = voyageai.Client(
            api_key=api_key,  # Or use VOYAGE_API_KEY environment variable
        )

    def as_embeddings(self, model: str = 'voyage-4'):
        """
        :param model: see in https://www.mongodb.com/docs/voyageai/models/#choosing-a-model
        """
        super().as_embeddings(model)

    def encode(self, *_input: str) -> list[list[float]]:
        result = self.client.embed(
            texts=list(_input),
            model=self.model
        )
        return result.embeddings
