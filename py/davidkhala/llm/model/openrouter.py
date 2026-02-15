from typing import Literal

from davidkhala.llm.model.chat import CompareChatAware
from davidkhala.llm.model.garden import GardenAlike


class OpenRouterModel(CompareChatAware, GardenAlike):
    def __init__(self):
        super().__init__()
        del self.n  # openrouter has no `n` parameter support TODO Is it global? or limited to openrouter?

    @property
    def free_models(self) -> list[str]:
        l = list(
            map(lambda model: model['id'],
                filter(lambda model: model['id'].endswith(':free'), self.list_models())
                )
        )
        l.append('openrouter/free')
        return l


class Plugins:
    PDF_ENGINE = Literal['pdf-text', 'mistral-ocr', 'native'] | None

    @staticmethod
    def pdf(engine: PDF_ENGINE = 'pdf-text'):
        """
        PDF parsing will still work even if the plugin is not explicitly set
        :param engine:
            'pdf-text': free. Best for well-structured PDFs with clear text content
            'native': Only available for models that support file input natively. Token cost
            'mistral-ocr': default. Best quality. $2 per 1,000 pages
        """
        return {'id': 'file-parser', 'pdf': {'engine': engine}}
