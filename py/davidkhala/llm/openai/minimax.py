from davidkhala.llm.model.chat import MessageDict
from davidkhala.llm.openai import Client as BaseClient, OpenAI, on_response


class Client(BaseClient):
    n = 1  # minimax does not support n

    def chat(self, *user_prompt, **kwargs) -> tuple[str, list[dict]]:
        kwargs['extra_body'] = {'reasoning_split': True}
        response = super().chat_create(*user_prompt, **kwargs)

        assert len(response.choices) == Client.n, "minimax does not support n"
        message = response.choices[0].message

        self.for_next(MessageDict(content=message.content, role=message.role))

        return message.content, message.reasoning_details  # reasoning_details is hidden field of minimax


class Global(Client):
    def __init__(self, api_key: str):
        super().__init__(OpenAI(api_key=api_key, base_url='https://api.minimax.io/v1'))


class CN(Client):
    def __init__(self, api_key: str):
        super().__init__(OpenAI(api_key=api_key, base_url='https://api.minimaxi.com/v1'))
