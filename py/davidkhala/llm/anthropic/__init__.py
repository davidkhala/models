from anthropic import Anthropic, __version__
from anthropic.types import Message

from davidkhala.llm.model.chat import ChatAware, Prompt
from davidkhala.llm.model.file import FileAware

version = __version__


class Client(ChatAware, FileAware):
    def __init__(self, *,
                 api_key: str | None = None,
                 base_url: str | None = None, ):
        super().__init__()
        self.client = Anthropic(api_key=api_key, base_url=base_url)
        self.system: str | None = None  # system prompt
        self.max_tokens = 1024

    def chat(self, *user_prompt: Prompt):

        r: Message = self.client.messages.create(
            model=self.model,
            system=self.system,
            messages=self.messages_from(*user_prompt),
            max_tokens=self.max_tokens,  # required
        )
        self.messages.append({
            'content': r.content,
            'role': r.role,
        })
        # TODO impl. DocumentBlockParam
        return [{'text': _.text, 'citations': _.citations} for _ in r.content if _.type == 'text']

    def as_chat(self, model: str | None, sys_prompt: str = None):
        self.model = model
        if sys_prompt is not None:
            self.system = sys_prompt
