from davidkhala.utils.http_request import default_on_response
from requests import Response, HTTPError as RawHTTPError

from davidkhala.llm.api import GardenAPI, ChatAPI, EmbeddingAPI
from davidkhala.llm.model.chat import ImagePrompt, ChoicesChat
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


class SiliconFlow(ChatAPI, ChoicesChat, Reranker, EmbeddingAPI, GardenAPI):
    @property
    def free_models(self) -> list[str]:
        """
        Cannot be lively fetched by list_models
        """
        return [
            # chat section
            'THUDM/GLM-4.1V-9B-Thinking'
            'THUDM/GLM-Z1-9B-0414'
            'THUDM/GLM-4-9B-0414'
            'THUDM/glm-4-9b-chat'
            'Qwen/Qwen3-8B'
            'Qwen/Qwen2.5-7B-Instruct'
            'Qwen/Qwen2.5-Coder-7B-Instruct'
            'internlm/internlm2_5-7b-chat'
            'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
            # embedding and reranker
            'BAAI/bge-m3'
            'BAAI/bge-reranker-v2-m3'
            'BAAI/bge-large-zh-v1.5'
            'BAAI/bge-large-en-v1.5'
            'netease-youdao/bce-reranker-base_v1'
            'netease-youdao/bce-embedding-base_v1'
            # Audio
            'FunAudioLLM/SenseVoiceSmall'
            # image
            'Kwai-Kolors/Kolors'
        ]

    def __init__(self, api_key: str):
        super().__init__(api_key, 'https://api.siliconflow.cn')

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
