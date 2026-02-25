import base64
import json
from pathlib import Path

from davidkhala.utils.syntax.compat import deprecated
from davidkhala.utils.syntax.format import mime_of
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from openai.types.chat import (
    ChatCompletionUserMessageParam, ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from davidkhala.llm.model.chat import on_response
from davidkhala.llm.openai import Client


class AzureHosted(Client):
    def chat(self, *user_prompt, **kwargs):
        if 'web_search_options' in kwargs:
            raise ValueError('Web search options not supported in any models of Azure AI Foundry')
        return super().chat(*user_prompt, **kwargs)


from davidkhala.ml.ocr.model import FieldProperties


class ModelDeploymentClient(AzureHosted):

    def __init__(self, key, deployment):
        super().__init__(AzureOpenAI(
            api_version="2024-12-01-preview",  # mandatory
            azure_endpoint=f"https://{deployment}.cognitiveservices.azure.com/",
            api_key=key,
        ))

    def process(self, file: Path, schema: dict[str, FieldProperties],
                *,
                prompt="Extract the required fields from this image and return the output strictly following the provided JSON schema.") \
            -> list[dict]:
        with open(file, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
        required = [k for k, _ in schema.items() if _.required]
        properties = {k: {'type': v.type} for k, v in schema.items()}

        json_schema = JSONSchema(
            name='-',
            schema={"type": "object",
                    "properties": properties,
                    "required": required,
                    },
        )

        self.messages.append(ChatCompletionUserMessageParam(
            role='user',
            content=[
                ChatCompletionContentPartTextParam(
                    type='text',
                    text=prompt),
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=f"data:{mime_of(file)};base64,{content}",
                        detail='auto'
                    )
                )
            ]
        ))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            response_format=ResponseFormatJSONSchema(
                type='json_schema',
                json_schema=json_schema
            ),
            n=self.n,
        )
        return [json.loads(_) for _ in on_response(response, self.n)]


@deprecated("Azure Open AI is deprecated. Please migrate to Microsoft Foundry")
class OpenAIClient(AzureHosted):

    def __init__(self, api_key, project):
        super().__init__(OpenAI(
            base_url=f"https://{project}.openai.azure.com/openai/v1/",
            api_key=api_key,
        ))

    def as_chat(self, model="gpt-oss-120b", sys_prompt: str = None):
        super().as_chat(model, sys_prompt)
