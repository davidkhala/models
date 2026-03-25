import json

from davidkhala.utils.http_request import default_on_response, Request
from requests import Response, HTTPError as RawHTTPError

from davidkhala.llm.api import GardenAPI, ChatAPI, EmbeddingAPI
from davidkhala.llm.model.chat import ImagePrompt, ChoicesChat
from davidkhala.llm.model.garden import TrialAvailable
from davidkhala.llm.model.rerank import Reranker

Prompt = str | ImagePrompt


class HTTPError(RawHTTPError):
    @staticmethod
    def from_raw(err: RawHTTPError):
        return HTTPError(err.args[0], response=err.response)

    @property
    def code(self):
        return self.json['code']

    @property
    def json(self):
        return self.response.json()

    @property
    def message(self):
        return self.json['message']


class CNConsole(Request, TrialAvailable):

    def __init__(self, token: str):
        super().__init__()
        self.open()
        self.session.cookies.set("__SF_auth.session-token", token)

    @property
    def free_models(self) -> list[str]:
        r = []
        for model, name in self.models:

            price_dict = self.price_of(model)
            if sum(price_dict.values()) == 0.0:
                r.append(name)
        return r

    @property
    def models(self) -> list[tuple[int, str]]:

        url = "https://siliconflow.cn/models"

        def on_response(response: Response) -> str | None:
            if response.ok:
                if response.text: return response.text
                return None
            else:
                return response.raise_for_status()

        self.on_response = on_response
        html = self.request(url, 'GET')
        self.on_response = default_on_response

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        scripts = soup.find_all("script")
        prefix = 'self.__next_f.push([1,"4'
        for s in scripts:
            if s.string and prefix in s.string:
                trim = s.string[len(prefix) + 1:-3]

                data = json.loads(trim.encode().decode('unicode_escape'))
                return [(_['modelId'], _['modelName']) for _ in data[1][3]['data']]
        assert False

    def price_of(self, model_id):
        url = f"https://cloud.siliconflow.cn/biz-server/api/v1/playground/{model_id}/biz_info"
        response = self.request(url, 'GET')
        functionCodes = ["input-tokens", "output-tokens", "image-cnt", "video-cnt", "cached-input-tokens", "utf8-bytes"]
        result = {}
        match response:
            case {'code': 20000, 'message': 'Ok', 'status': True, 'data': data}:
                groups = data["definitionPricingGroup"][0]["functionPricingGroups"]
                for g in groups:
                    code = g["functionInfo"]["functionCode"]
                    if code in functionCodes:
                        result[code] = float(g["pricingGroups"][0]["pricingInfo"]["unitPriceAmount"])
                    else:
                        assert False, f"new functionCode found: {code}"

                return result


class SiliconFlow(ChatAPI, ChoicesChat, Reranker, EmbeddingAPI, GardenAPI):

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)

        def on_response(response: Response):
            try:
                return default_on_response(response)
            except RawHTTPError as e:
                raise HTTPError.from_raw(e)

        self.on_response = on_response

    def chat(self, *user_prompt: Prompt):
        r = super().chat(*user_prompt, model=self.model, timeout=50, n=self.n)

        data = r['data']
        assert len(data) == self.n
        # siliconflow has no data structure in choices[0].message.annotations
        # See in https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions#response-choices-items-message
        assert len(r['annotations']) == 0
        #
        return data

    def which(self, query: str, documents: list[str], **kwargs) -> tuple[str, int]:
        json = {
            'model': self.model,
            'query': query,
            'documents': documents,
            **kwargs
        }
        response = self.request(f"{self.base_url}/rerank", "POST", json=json)
        most_relevant_index = max(response['results'], key=lambda x: x['relevance_score'])['index']

        return documents[most_relevant_index], most_relevant_index


class CN(SiliconFlow):
    def __init__(self, api_key: str):
        super().__init__(api_key, 'https://api.siliconflow.cn')


class Global(SiliconFlow):
    def __init__(self, api_key: str):
        super().__init__(api_key, 'https://api.siliconflow.com')
