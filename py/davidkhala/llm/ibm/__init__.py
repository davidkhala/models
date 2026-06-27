from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from davidkhala.llm.model.chat import ChoicesChat, DeterministicChat, ChoicesAware
from davidkhala.llm.model.prompt.param import Prompt


class Client(ChoicesChat, DeterministicChat):
    def __init__(self, project_id, *, region: str, api_key: str):
        super().__init__()
        self.client = APIClient(Credentials(
            url=f"https://{region}.ml.cloud.ibm.com",
            api_key=api_key
        ), project_id)

    def as_chat(self, model: str | None, sys_prompt: str = None):
        super().as_chat(model=None, sys_prompt=sys_prompt)
        self.handler = ModelInference(
            model_id=model,
            api_client=self.client,
        )

    def chat(self, *user_prompt: Prompt):
        response: dict = self.handler.chat(
            messages=self.messages_from(*user_prompt),
            params=TextChatParameters(
                seed=self.seed,
                n=self.n
            )
        )
        return ChoicesAware.model_validate(response)
