from davidkhala.llm.model.chat import ChatAware, on_response
from davidkhala.llm.zai import Client as BaseClient


class Client(ChatAware, BaseClient):
    n = 1

    def chat(self, *user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages_from(*user_prompt),
        )
        print(response)
        # z.ai has no `n` support
        return on_response(response, Client.n)
