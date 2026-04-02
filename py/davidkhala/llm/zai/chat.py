from zai.types.chat import Completion

from davidkhala.llm.model.chat import ChatAware
from davidkhala.llm.zai import GlobalClient as BaseClient


class GlobalClient(ChatAware, BaseClient):

    def chat(self, *user_prompt) -> str | None:
        """
            z.ai has no `n` support
            z.ai has no `seed` support. Knowing that SDK provides `seed` option, it does not guarantee determinism
        """
        response: Completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages_from(*user_prompt),
        )
        return response.choices[0].message.content
