from google import genai
from google.genai.types import Candidate, GenerateContentResponse, GenerateContentConfig
from pydantic import BaseModel

from davidkhala.llm.model.chat import ChoicesChat


class ChoicesAware(BaseModel):
    candidates: list[Candidate]


def from_api_key(api_key: str, *, project: str | None = None) -> genai.Client:
    """
    :param api_key: API Restrictions: `Agent Platform API`
    :param project: gcp project ID
    """
    return genai.Client(
        enterprise=True if project else False,
        api_key=api_key,
        project=project,
        location="global"
    )
class Client(ChoicesChat):

    def __init__(self, client: genai.Client):
        super().__init__()
        self.client = client

    def chat(self, *user_prompt, **kwargs):
        r = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=GenerateContentConfig(candidate_count=self.n)
        )
        return self.on_response(r)

    def on_response(self, response: GenerateContentResponse) -> list:
        contents = [choice.content for choice in response.candidates]
        assert len(contents) == self.n, f"expected {self.n} choices, but got {len(contents)}"
        return contents
