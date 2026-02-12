import time

import requests
from davidkhala.utils.http_request import default_on_response
from requests import Response

from davidkhala.llm.api import API
from davidkhala.llm.model.chat import CompareChatAware


class OpenRouter(API, CompareChatAware):

    @property
    def free_models(self) -> list[str]:
        l = list(
            map(lambda model: model['id'],
                filter(lambda model: model['id'].endswith(':free'), self.list_models())
                )
        )
        l.append('openrouter/free')
        return l

    @staticmethod
    def on_response(response: requests.Response):
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

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, 'https://openrouter.ai/api')

        if 'leaderboard' in kwargs and type(kwargs['leaderboard']) is dict:
            self.options["headers"]["HTTP-Referer"] = kwargs['leaderboard'][
                'url']  # Site URL for rankings on openrouter.ai.
            self.options["headers"]["X-Title"] = kwargs['leaderboard'][
                'name']  # Site title for rankings on openrouter.ai.
        self.on_response = OpenRouter.on_response
        self.retry = True

    def request(self, url, method: str, params=None, data=None, json=None) -> dict:
        try:
            return super().request(url, method, params, data, json)
        except requests.HTTPError as e:
            if e.response.status_code == 429 and self.retry:  # 429: You are being rate limited
                time.sleep(1)
                return self.request(url, method, params, data, json)
            else:
                raise

    def as_chat(self, *models: str, sys_prompt: str = None):
        CompareChatAware.as_chat(self, *models, sys_prompt=sys_prompt)

    def chat(self, *user_prompt: str, **kwargs):
        if self._models:
            kwargs["models"] = self._models
        else:
            kwargs["model"] = self.model

        r = super().chat(*user_prompt, **kwargs)

        if self._models:
            assert r['model'] in self._models
        return r
