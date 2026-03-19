from openai import omit
from openai.types.chat import ChatCompletion

from davidkhala.llm.model.chat import on_response, ChoicesChat, DeterministicChat
from davidkhala.llm.openai import Client as BaseClient


class Client(BaseClient, ChoicesChat, DeterministicChat):

    def chat_create(self, *user_prompt, **kwargs) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages_from(*user_prompt),
            n=self.n,
            seed=self.seed or omit,
            **kwargs
        )

    def chat(self, *user_prompt, **kwargs):
        response: ChatCompletion = self.chat_create(*user_prompt, **kwargs)
        return on_response(response, self.n)
