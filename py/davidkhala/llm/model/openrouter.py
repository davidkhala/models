from typing import Literal

from davidkhala.llm.model.chat import ChatAware


class OpenRouterModel(ChatAware):
    n = 1  # openrouter has no `n` parameter support, only has one fake choice. Openrouter use models as pool for load-balance only

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._models: list[str] = []

    def as_chat(self, *models: str, sys_prompt: str = None):
        # migrate from CompareChatAware # TODO, any other openrouter alter
        if len(models) > 1:
            self._models = list(models)
            super().as_chat(None, sys_prompt)
        elif len(models) == 1:
            super().as_chat(models[0], sys_prompt)


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
