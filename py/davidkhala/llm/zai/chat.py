from davidkhala.llm.model.chat import ChatAware, on_response
from davidkhala.llm.zai import GlobalClient as BaseClient


class GlobalClient(ChatAware, BaseClient):
    n = 1

    def chat(self, *user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages_from(*user_prompt),
        )
        # z.ai has no `n` support
        # z.ai has no `seed` support. Knowing that SDK provides `seed` option, it does not guarantee determinism
        return on_response(response, GlobalClient.n)[0]
