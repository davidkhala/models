from openai import OpenAI
from openai.resources.responses import Responses
from openai.types.responses import Response

from davidkhala.llm.model.chat import ChatAware
from davidkhala.llm.openai import Client as BaseClient


class Client(BaseClient, ChatAware):
    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.instructions: str | None = None

    def as_chat(self, model: str | None, sys_prompt: str = None):
        self.model = model
        if sys_prompt is not None:
            self.instructions = sys_prompt

    def chat(self, *user_prompt, **kwargs) -> Response:
        responses: Responses = self.client.responses
        response: Response = responses.create(
            model=self.model,
            input=self.messages_from(*user_prompt),
            instructions=self.instructions,
            **kwargs
        )
        return response

    def with_annotations(self, **kwargs):
        raise NotImplementedError("annotations is not supported")
