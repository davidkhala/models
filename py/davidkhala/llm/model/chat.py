from abc import abstractmethod, ABC
from typing import Protocol, Any, TypeAlias

from pydantic import BaseModel

from davidkhala.llm.model import ModelAware
from davidkhala.llm.model.prompt import TextContentPart
from davidkhala.llm.model.prompt.annotation import ContentPart as AnnotationContent
from davidkhala.llm.model.prompt.file import ContentPart as FileContent
from davidkhala.llm.model.prompt.image import ContentPart as ImageContent
from davidkhala.llm.model.prompt.param import Prompt, Image, File

ContentPart = FileContent | ImageContent | TextContentPart | AnnotationContent


class MessageProtocol(Protocol):
    content: str | list | Any | None
    role: str


class ChoiceProtocol(Protocol):
    message: MessageProtocol


class ChoicesAware(Protocol):
    choices: list[ChoiceProtocol]


def on_response(response: ChoicesAware, n: int | None):
    contents = [choice.message.content for choice in response.choices]
    if n:
        assert len(contents) == n, f"expected {n} choices, but got {len(contents)}"
    return contents


MessageDict: TypeAlias = dict


class Message(BaseModel):
    content: str | list[ContentPart] = []
    role: str
    annotations: list[AnnotationContent] | None = None

    def as_dict(self) -> MessageDict:
        return self.model_dump()

    @staticmethod
    def from_dict(data: MessageDict):
        return Message.model_validate(data)


def message_from(*user_prompt: Prompt) -> Message:
    message = Message(role='user')
    for _ in user_prompt:
        match _:
            case str():
                message.content.append(TextContentPart(text=_))
            case Image() | File():
                message.content.append(_.expand())

    return message


class ChatAware(ModelAware, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages: list[dict | MessageDict] = []

    def reset(self):
        self.messages = []

    def as_chat(self, model: str | None, sys_prompt: str = None):
        self.model = model
        if sys_prompt is not None:
            self.messages = [Message(role='system', content=sys_prompt).as_dict()]

    @abstractmethod
    def chat(self, *user_prompt, **kwargs):
        ...

    def messages_from(self, *user_prompt: Prompt):
        self.messages.append(message_from(*user_prompt).as_dict())
        return self.messages

    def for_next(self, message: Message | MessageDict):
        match message:
            case Message():
                self.messages.append(message.as_dict())
            case dict():
                self.messages.append(message)

    def with_annotations(self, annotations: list[AnnotationContent]):
        """In classic openai and [openrouter](https://openrouter.ai/docs/guides/overview/multimodal/pdfs#skip-parsing-costs)"""
        # file should not be excluded from message thread. Just openrouter will skip parsing costs
        self.for_next(Message(role="assistant", annotations=annotations).as_dict())


class ChoicesChat(ChatAware, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n: int = 1

    def reset(self):
        super().reset()
        self.n = 1


class DeterministicChat:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1  # random seed
