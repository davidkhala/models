import datetime

from davidkhala.utils.http_request import Request

from davidkhala.llm.model.chat import ChatAware, Prompt
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.garden import GardenAlike, ID


class API(Request, ChatAware, EmbeddingAware, GardenAlike):
    def __init__(self, api_key: str, base_url: str):
        ChatAware.__init__(self)
        Request.__init__(self, {
            "bearer": api_key
        })
        self.base_url = base_url + '/v1'

    def chat(self, *user_prompt: Prompt, **kwargs):
        json = {
            "messages": self.messages_from(*user_prompt),
            **kwargs,
        }

        response = self.request(f"{self.base_url}/chat/completions", "POST", json=json)

        return {
            "data": list(map(lambda x: x['message']['content'], response['choices'])),
            "meta": {
                "usage": response['usage'],
                "created": datetime.datetime.fromtimestamp(response['created'])
            },
            'model': response['model'],
        }

    def encode(self, *_input: str) -> list[list[float]]:
        response = self.request(f"{self.base_url}/embeddings", "POST", json={
            'input': _input,
            'model': self.model
        })
        return [_['embedding'] for _ in response['data']]

    def list_models(self)->list[ID]:
        response = self.request(f"{self.base_url}/models", "GET")
        return response['data']
