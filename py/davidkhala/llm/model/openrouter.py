from abc import ABC
from typing import Literal

from davidkhala.llm.model.chat import CompareChatAware
from davidkhala.llm.model.garden import GardenAlike


class OpenRouterModel(CompareChatAware,GardenAlike, ABC):
    """
    openrouter has no `n` parameter support
    """
    n = 1  # only has one fake choice. Openrouter use models as pool for load-balance only


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
