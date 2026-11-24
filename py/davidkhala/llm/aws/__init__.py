from pathlib import PurePath
from typing import Literal

import boto3
from davidkhala.utils.syntax.format import sniff_image_format
from pydantic import BaseModel

from davidkhala.llm.model.chat import Message, ChatAware
from davidkhala.llm.model.prompt.param import Prompt, Image, File


class TextContentPart(BaseModel):
    text: str


class SourceBytes(BaseModel):
    bytes: bytes


class ImageBlock(BaseModel):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: SourceBytes


class DocumentBlock(BaseModel):
    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str
    source: SourceBytes


class ImageContentPart(BaseModel):
    image: ImageBlock

    @staticmethod
    def from_prompt(prompt: Image):
        data = prompt.source()
        image_format = sniff_image_format(data)
        if image_format is None:
            raise ValueError("Unsupported or unrecognized image format")
        return ImageContentPart(
            image=ImageBlock(format=image_format, source=SourceBytes(bytes=data))
        )


class FileContentPart(BaseModel):
    document: DocumentBlock

    @staticmethod
    def from_prompt(prompt: File):
        filename = PurePath(prompt.filename)
        return FileContentPart(
            document=DocumentBlock(
                format=filename.suffix.lstrip(".").lower(),
                name=filename.stem,
                source=SourceBytes(bytes=prompt.source()),
            )
        )


class Client(ChatAware):
    def __init__(
            self,
            region="ap-southeast-1",
            access_key_id: str | None = None,
            secret_access_key: str | None = None,
    ):
        """
        If credentials are omitted, boto3's default credential chain applies (environment variables, profile, etc.).
        """
        super().__init__()
        self.region = region
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def as_chat(self, model: str, sys_prompt: str | None = None):
        self.model = f"global.{model}"
        if sys_prompt is not None:
            self.messages = [Message(role='system', content=sys_prompt).as_dict()]

    def chat(self, *user_prompt):
        """
        https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-runtime/client/converse.html
        """
        response = self.client.converse(
            modelId=self.model,
            messages=self.messages_from(*user_prompt),
        )
        return self.on_response(response)

    def messages_from(self, *user_prompt: Prompt):
        content = []
        for _ in user_prompt:
            match _:
                case str():
                    content.append(TextContentPart(text=_))
                case Image():
                    content.append(ImageContentPart.from_prompt(_))
                case File():
                    content.append(FileContentPart.from_prompt(_))

        self.messages.append(Message(role='user', content=content).as_dict())
        return self.messages

    def on_response(self, response) -> list:
        return [choice['text'] for choice in response["output"]["message"]["content"] if "text" in choice ]

