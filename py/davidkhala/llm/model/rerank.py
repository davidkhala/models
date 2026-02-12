from davidkhala.llm.model import ModelAware


class Reranker(ModelAware):
    def which(self, query: str, documents: list[str], **kwargs) -> tuple[str, int]:
        """
        Returns:
            tuple:
                document (str): The most relevant document
                index (int): The index of most relevant document in documents

        """
        ...