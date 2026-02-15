import time

import requests
from davidkhala.utils.http_request import default_on_response
from requests import Response

from davidkhala.llm.api import API
from davidkhala.llm.model.chat import Prompt
from davidkhala.llm.model.openrouter import OpenRouterModel, Plugins


class OpenRouter(API, OpenRouterModel):

    def __init__(self, api_key: str, **kwargs):
        API.__init__(self, api_key, 'https://openrouter.ai/api')
        OpenRouterModel.__init__(self)

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

    def request(self, url, method: str, params=None, data=None, json=None) -> dict:
        try:
            return super().request(url, method, params, data, json)
        except requests.HTTPError as e:
            if e.response.status_code == 429 and self.retry:  # 429: You are being rate limited
                time.sleep(1)
                return self.request(url, method, params, data, json)
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
        assert len(data) == 1  # only has one answer. Openrouter use models as pool for load-balance only
        if _a:
            assert len(_a) == 1
            self.with_annotations(_a[0])
        if self._models:
            assert r['model'] in self._models
        return data[0]

    @property
    def models(self) -> list[str]:
        return [m['id'] for m in self.list_models()]
