from typing import Protocol, Any, Iterable, TypedDict

from davidkhala.llm.model import ModelAware


class MessageProtocol(Protocol):
    content: str | Any


class ChoiceProtocol(Protocol):
    message: MessageProtocol


class ChoicesAware(Protocol):
    choices: list[ChoiceProtocol]


class ImagePromptDict(TypedDict):
    text: str
    image_url: list[str]


def on_response(response: ChoicesAware, n: int | None):
    contents = [choice.message.content for choice in response.choices]
    if n:
        assert len(contents) == n, f"expected {n} choices, but got {len(contents)}"
    return contents


class MessageDict(TypedDict):
    content: str | list | None
    role: str


def messages_from(*user_prompt: str | ImagePromptDict) -> Iterable[MessageDict]:
    for _ in user_prompt:
        message = MessageDict(role='user', content=None)
        if type(_) == str:
            message['content'] = _
        elif type(_) == dict:
            message['content'] = [{"type": "text", "text": _['text']}]
            message['content'].extend({"type": "image_url", "image_url": {"url": i}} for i in _['image_url'])
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
        messages = list(self.messages)
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
