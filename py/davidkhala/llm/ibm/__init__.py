from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from davidkhala.llm.model.chat import ChatAware


class Client(ChatAware):
    def __init__(self, project_id, *, base_url: str, api_key: str):
        super().__init__()
        self.client = APIClient(Credentials(
            url = base_url,
            api_key = api_key
        ))
        self.project_id = project_id

    def as_chat(self, model: str | None, sys_prompt: str = None):
        super().as_chat(model=None, sys_prompt=sys_prompt)
        self.handler = ModelInference(
            model_id=model,
            api_client=self.client,
            project_id=self.project_id,
        )
    def chat(self, *messages:str):
        # TODO test cover
        return self.handler.chat(messages=self.messages_from(messages))



