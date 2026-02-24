from openai import OpenAI
from davidkhala.llm.openai import Client as BaseClient


class Client(BaseClient):
    def __init__(self, o: str | int, token: str):
        """
        :param o: databricks organization id. Appears as parameter of workspace url
        """
        super().__init__(OpenAI(
            base_url=f"https://{o}.ai-gateway.cloud.databricks.com/mlflow/v1",
            api_key=token
        ))

    def chat(self, *user_prompt, **kwargs):
        """Databricks always reasoning, and content is not of type str"""
        contents:list[list[dict]] = super().chat(*user_prompt, **kwargs)

        for content in contents:
            for block in content:
                match block['type']:
                    case "text":
                        yield block['text']
                    case "reasoning":
                        for s in block["summary"]:
                            assert s['type'] == 'summary_text'
                            yield s['text']