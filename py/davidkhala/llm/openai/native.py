from typing import Literal

from openai import OpenAI
from openai.types.responses import ToolParam, WebSearchToolParam

from davidkhala.llm.openai.current import Client


class NativeClient(Client):
    def __init__(self, api_key, base_url=None):
        super().__init__(OpenAI(
            api_key=api_key,
            base_url=base_url
        ))

    def chat(self, user_prompt, *, web_search: Literal["low", "medium", "high"] = None):
        tools: list[ToolParam] = []
        if web_search:
            tools.append(WebSearchToolParam(
                type="web_search",
                search_context_size=web_search,
            ))
        return super().chat(user_prompt, tools=tools)
