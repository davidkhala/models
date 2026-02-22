import datetime

from davidkhala.utils.http_request import Request

from davidkhala.llm.model.chat import ChatAware, Prompt
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.garden import GardenAlike, ID


class API(Request):
    def __init__(self, api_key: str, base_url: str):
        super().__init__({
            "bearer": api_key
        })
        self.base_url = base_url + '/v1'


class ChatAPI(API, ChatAware):

    def chat(self, *user_prompt: Prompt, **kwargs):
        self.messages_from(*user_prompt)
        json = {
            "messages": self.messages,
            **kwargs,
        }

        response = self.request(f"{self.base_url}/chat/completions", "POST", json=json)

        data = []
        file_annotations = []
        for _ in response['choices']:
            data.append(_['message']['content'])
            _a = _["message"].get('annotations')
            if _a: file_annotations.append(_a)

        return {
            "data": data,
            "meta": {
                "usage": response['usage'],
                "created": datetime.datetime.fromtimestamp(response['created']),
            },
            'model': response['model'],
            'annotations': file_annotations,
        }


class EmbeddingAPI(API, EmbeddingAware):
    def encode(self, *_input: str) -> list[list[float]]:
        response = self.request(f"{self.base_url}/embeddings", "POST", json={
            'input': _input,
            'model': self.model
        })
        return [_['embedding'] for _ in response['data']]


class GardenAPI(API, GardenAlike):
    def list_models(self) -> list[ID]:
        response = self.request(f"{self.base_url}/models", "GET")
        return response['data']
