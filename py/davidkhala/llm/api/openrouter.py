from time import sleep

import requests
from davidkhala.utils.http_request import default_on_response
from requests import Response

from davidkhala.llm.api import EmbeddingAPI, ChatAPI
from davidkhala.llm.model.chat import Prompt
from davidkhala.llm.model.openrouter import OpenRouterModel, Plugins


class OpenRouter(ChatAPI, EmbeddingAPI, OpenRouterModel):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key=api_key, base_url='https://openrouter.ai/api')

        if 'leaderboard' in kwargs and type(kwargs['leaderboard']) is dict:
            # Site URL for rankings on openrouter.ai.
            self.options["headers"]["HTTP-Referer"] = kwargs['leaderboard']['url']
            # Site title for rankings on openrouter.ai.
            self.options["headers"]["X-Title"] = kwargs['leaderboard']['name']
        self.retry = True

        def on_response(response: requests.Response):
            """

            """
            r = default_on_response(response)
            # openrouter special error on response.ok
            err = r.get('error')
            if err:
                derived_response = Response()
                derived_response.status_code = err['code']
                derived_response.reason = err['message']
                derived_response.metadata = err.get("metadata")

                derived_response.raise_for_status()
            return r

        self.on_response = on_response

    def request(self, *args, **kwargs) -> dict:
        try:
            return super().request(*args, **kwargs)
        except requests.HTTPError as e:
            if e.response.status_code == 429 and self.retry:  # 429: You are being rate limited
                sleep(1)
                return self.request(*args, **kwargs)
            else:
                raise

    def chat(self, *user_prompt: Prompt, pdf_engine: Plugins.PDF_ENGINE = 'pdf-text'):
        options: dict = {
            'plugins': [
                Plugins.pdf(pdf_engine)
            ]
        }
        if self._models:
            options["models"] = self._models
        else:
            options["model"] = self.model

        r = super().chat(*user_prompt, **options)

        data, _a = r['data'], r['annotations']
        assert len(data) == OpenRouter.n
        assert len(_a) <= OpenRouter.n
        if _a:
            self.with_annotations(_a[0])
        if self._models:
            assert r['model'] in self._models
        return data[0]

    @property
    def models(self) -> list[str]:
        return [m['id'] for m in self.list_models()]
