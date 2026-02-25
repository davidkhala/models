from davidkhala.llm.anthropic import Client as AnthropicClient


class Global(AnthropicClient):
    def __init__(self, api_key:str):
        super().__init__(api_key=api_key, base_url='https://api.minimax.io/anthropic')
class CN(AnthropicClient):
    def __init__(self, api_key:str):
        super().__init__(api_key=api_key, base_url='https://api.minimaxi.com/anthropic')