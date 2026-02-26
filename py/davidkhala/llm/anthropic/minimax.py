from davidkhala.llm.anthropic import Client as AnthropicClient
from davidkhala.llm.model.chat import Prompt


class Client(AnthropicClient):
    def chat(self, *user_prompt: Prompt):
        """
        minimax supports text and tool calls, no image/document input
        """
        return super().chat(*user_prompt)


class Global(Client):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url='https://api.minimax.io/anthropic')


class CN(Client):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url='https://api.minimaxi.com/anthropic')
