from openai import OpenAI
from davidkhala.llm.openai import Client as BaseClient


class Client(BaseClient):
    def __init__(self, host: str, token: str):
        super().__init__(OpenAI(
            base_url=f"https://{host}/serving-endpoints",
            api_key=token
        ))

    def chat(self, *user_prompt, **kwargs):
        """Databricks always reasoning"""
        rs = super().chat(*user_prompt, **kwargs)
        for r in rs:
            assert len(r) == 2
            assert r[0]['type'] == 'reasoning'
            for s in r[0]['summary']:
                assert s['type'] == 'summary_text'
                yield s['text']
            assert r[1]['type'] == 'text'
            yield r[1]['text']
