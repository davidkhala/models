from pathlib import Path
from typing import Protocol, Any, Iterable, TypedDict

from pydantic import BaseModel
from davidkhala.utils.syntax.format import Base64
from davidkhala.utils.syntax.url import filename_from
from davidkhala.llm.model import ModelAware


class MessageProtocol(Protocol):
    content: str | Any


class ChoiceProtocol(Protocol):
    message: MessageProtocol


class ChoicesAware(Protocol):
    choices: list[ChoiceProtocol]


class MultimodalPrompt(BaseModel):
    text: str


class ImagePrompt(MultimodalPrompt):
    image_url: list[str]


class FilePrompt(MultimodalPrompt):
    url: str | None = None # TODO can it be multiple like urls:list[str]?
    path: Path | None = None # either by url or path


Prompt = str | ImagePrompt | FilePrompt


def on_response(response: ChoicesAware, n: int | None):
    contents = [choice.message.content for choice in response.choices]
    if n:
        assert len(contents) == n, f"expected {n} choices, but got {len(contents)}"
    return contents


class MessageDict(TypedDict):
    content: str | list | None
    role: str


def messages_from(*user_prompt: Prompt) -> Iterable[MessageDict]:
    for _ in user_prompt:
        message = MessageDict(role='user', content=None)
        match _:
            case str():
                message['content'] = _
            case MultimodalPrompt():
                message['content'] = [{"type": "text", "text": _.text}]
                match _:
                    case ImagePrompt():
                        message['content'].extend({"type": "image_url", "image_url": {"url": i}} for i in _.image_url)
                    case FilePrompt():
                        if _.url:
                            _filename = filename_from(_.url)
                            url = _.url
                        else:
                            _filename = _.path.name
                            url = f"data:application/pdf;base64,{Base64.encode_file(_.path)}"
                        message['content'].extend([{"type": "file", "file": {
                            "filename": _filename, "file_data": url
                        }}])

        yield message


class ChatAware(ModelAware):
    def __init__(self):
        super().__init__()
        self.messages: list[Any | MessageDict] = []
        self.n: int = 1

    def as_chat(self, model: str | None, sys_prompt: str = None):
        self.model = model
        if sys_prompt is not None:
            self.messages = [MessageDict(role='system', content=sys_prompt)]

    def chat(self, *user_prompt, **kwargs): ...

    def messages_from(self, *user_prompt) -> list[MessageDict]:
        messages = list(self.messages)  # clone a copy
        messages.extend(messages_from(*user_prompt))
        return messages


class CompareChatAware(ChatAware):
    def __init__(self):
        super().__init__()
        self._models = None

    def as_chat(self, *models: str, sys_prompt: str = None):
        if len(models) > 1:
            self._models = models
            super().as_chat(None, sys_prompt)
        elif len(models) == 1:
            super().as_chat(models[0], sys_prompt)
