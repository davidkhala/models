from openai import OpenAI
from davidkhala.llm.openai.classic import Client as BaseClient


class Client(BaseClient):
    def __init__(self, o: str | int, token: str):
        """
        :param o: databricks organization id. Appears as parameter of workspace url
        """
        super().__init__(OpenAI(
            base_url=f"https://{o}.ai-gateway.cloud.databricks.com/mlflow/v1",
            api_key=token
        ))
        self.seed = None  # databricks has no seed support

    def chat(self, *user_prompt, **kwargs) -> tuple[list[str], list[str]]:
        """Databricks always reasoning, and content is not of type str"""
        contents: list[list[dict] | str] = super().chat(*user_prompt, **kwargs)

        response = []
        reason = []
        for content in contents:
            if type(content) == str:
                response.append(content)
            for block in content:
                match block:
                    case {"type": "text", "text": text}:
                        response.append(text)
                    case {"type": "reasoning", "summary": summary}:
                        for s in summary:
                            if s.get("type") == "summary_text":
                                reason.append(s["text"])
        return response, reason
