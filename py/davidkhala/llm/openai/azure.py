import base64
import json
from pathlib import Path

from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from openai.types.responses import ResponseTextConfigParam, ResponseFormatTextJSONSchemaConfigParam, \
    EasyInputMessageParam, ResponseInputTextParam, ResponseInputFileParam

from davidkhala.llm.openai.current import Client


class AzureHosted(Client):
    def chat(self, user_prompt, **kwargs):
        if 'web_search' in kwargs:
            raise ValueError('Web search options not supported in any models of Azure AI Foundry')
        return super().chat(user_prompt, **kwargs)


from davidkhala.ml.ocr.model import FieldProperties


class ModelDeploymentClient(AzureHosted):

    def __init__(self, key, deployment):
        super().__init__(AzureOpenAI(
            api_version="2025-03-01-preview",  # mandatory
            azure_endpoint=f"https://{deployment}.cognitiveservices.azure.com/",
            api_key=key,
        ))

    def process(self, file: Path, schema: dict[str, FieldProperties],
                *,
                prompt="Extract the required fields from this file and return the output strictly following the provided JSON schema.",
                ) -> list[dict]:
        with open(file, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
        required = [k for k, _ in schema.items() if _.required]
        properties = {k: {'type': v.type} for k, v in schema.items()}

        self.messages.append(EasyInputMessageParam(
            role='user',
            content=[
                ResponseInputTextParam(
                    type='input_text',
                    text=prompt),
                ResponseInputFileParam(
                    type="input_file",
                    file_data=content,
                    filename=file.name,
                )
            ]
        ))
        json_schema = ResponseFormatTextJSONSchemaConfigParam(
            name='-', type="json_schema",
            schema={"type": "object",
                    "properties": properties,
                    "required": required,
                    },
        )
        response = self.chat(
            text=ResponseTextConfigParam(format=json_schema)
        )
        return json.loads(response)


class OpenAIClient(AzureHosted):

    def __init__(self, api_key, project):
        super().__init__(OpenAI(
            base_url=f"https://{project}.openai.azure.com/openai/v1/",
            api_key=api_key,
        ))
