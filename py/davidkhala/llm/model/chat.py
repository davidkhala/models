from abc import abstractmethod, ABC
from typing import Any, TypeAlias

from pydantic import BaseModel

from davidkhala.llm.model import ModelAware
from davidkhala.llm.model.prompt import TextContentPart
from davidkhala.llm.model.prompt.file import ContentPart as FileContent
from davidkhala.llm.model.prompt.image import ContentPart as ImageContent
from davidkhala.llm.model.prompt.param import Prompt, Image, File

ContentPart = FileContent | ImageContent | TextContentPart

MessageDict: TypeAlias = dict


class Message(BaseModel):
    content: str | list[ContentPart | Any] = []
    role: str
    annotations: list | None = None
    """
    @deprecated, annotations has been moved as an item in content list
    """

    def as_dict(self) -> MessageDict:
        return self.model_dump()

    @staticmethod
    def from_dict(data: MessageDict):
        return Message.model_validate(data)


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
        content = []
        for _ in user_prompt:
            match _:
                case str():
                    content.append(TextContentPart(text=_))
                case Image():
                    from davidkhala.llm.model.prompt.image import URL, ContentPart

                    content.append(ContentPart(image_url=URL(url=_.image_url, detail=_.detail)))
                case File():
                    from davidkhala.llm.model.prompt.file import ContentPart, File as _File

                    content.append(ContentPart(file=_File(filename=_.filename, file_data=_.file_data)))

        self.messages.append(Message(role='user', content=content).as_dict())
        return self.messages

    def for_next(self, message: Message | MessageDict):
        match message:
            case Message():
                self.messages.append(message.as_dict())
            case dict():
                self.messages.append(message)


class ChoiceModel(BaseModel):
    message: Message


class ChoicesAware(BaseModel):
    choices: list[ChoiceModel]


class ChoicesChat(ChatAware, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n: int = 1

    def reset(self):
        super().reset()
        self.n = 1

    def on_response(self, response) -> list:
        contents = [choice.message.content for choice in response.choices]
        assert len(contents) == self.n, f"expected {self.n} choices, but got {len(contents)}"
        return contents


class DeterministicChat:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1  # random seed
