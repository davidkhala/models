from abc import abstractmethod, ABC
from pathlib import Path
from typing import Protocol, Any, Iterable, TypedDict, Literal, NotRequired

from davidkhala.utils.syntax.format import Base64
from davidkhala.utils.syntax.url import filename_from
from pydantic import BaseModel

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
    detail: Literal['auto', 'low', 'high'] | None = 'auto'  # Image detail level for vision models


class FilePrompt(MultimodalPrompt):
    url: list[str] | None = None
    path: list[Path] | None = None


Prompt = str | ImagePrompt | FilePrompt


def on_response(response: ChoicesAware, n: int | None):
    contents = [choice.message.content for choice in response.choices]
    if n:
        assert len(contents) == n, f"expected {n} choices, but got {len(contents)}"
    return contents


class ImageContentPart(TypedDict):
    type: Literal['image_url']
    image_url: NotRequired[dict]


class TextContentPart(TypedDict):
    type: Literal['text']
    text: str


ContentPart = TextContentPart | ImageContentPart


class FileAnnotation(TypedDict):
    hash: str
    name: str | None
    content: list[ContentPart]


class AnnotationDict(TypedDict):
    type: Literal['file']
    file: FileAnnotation


class MessageDict(TypedDict):
    content: NotRequired[str | list]
    role: str
    annotations: NotRequired[list[AnnotationDict]]


import mimetypes


def messages_from(*user_prompt: Prompt) -> Iterable[MessageDict]:
    for _ in user_prompt:
        message = MessageDict(role='user')
        match _:
            case str():
                message['content'] = _
            case MultimodalPrompt():
                message['content'] = [{"type": "text", "text": _.text}]
                match _:
                    case ImagePrompt():
                        message['content'].extend(
                            {"type": "image_url", "image_url": {"url": i, 'detail': _.detail}} for i in _.image_url)

                    case FilePrompt():
                        # TODO Document Prompt in Anthropic don't align with this structure.
                        if _.url:
                            message['content'].extend({"type": "file", "file": {
                                "filename": filename_from(item), "file_data": item
                            }} for item in _.url)

                        if _.path:
                            for item in _.path:
                                mime, _ = mimetypes.guess_type(item)
                                message['content'].append({"type": "file", "file": {
                                    "filename": item.name,
                                    "file_data": f"data:{mime};base64,{Base64.encode_file(item)}"
                                }})

        yield message


class ChatAware(ABC, ModelAware):
    def __init__(self):
        super().__init__()
        self.messages: list[Any | MessageDict] = []

    def reset(self):
        self.messages = []

    def as_chat(self, model: str | None, sys_prompt: str = None):
        self.model = model
        if sys_prompt is not None:
            self.messages = [MessageDict(role='system', content=sys_prompt)]

    @abstractmethod
    def chat(self, *user_prompt, **kwargs): ...

    def messages_from(self, *user_prompt: Prompt):
        self.messages.extend(messages_from(*user_prompt))

    def with_annotations(self, annotations: list[AnnotationDict]):
        # file should not be excluded from message thread. Just openrouter will skip parsing costs
        self.messages.append({
            "role": "assistant",
            "annotations": annotations,
        })


class ChoicesChat(ChatAware, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n: int = 1

    def reset(self):
        super().reset()
        self.n = 1


class CompareChatAware(ChatAware, ABC):
    def __init__(self):
        super().__init__()
        self._models: list[str] = []

    def as_chat(self, *models: str, sys_prompt: str = None):
        if len(models) > 1:
            self._models = list(models)
            super().as_chat(None, sys_prompt)
        elif len(models) == 1:
            super().as_chat(models[0], sys_prompt)
