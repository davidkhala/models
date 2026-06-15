from pathlib import Path

from davidkhala.llm.model.chat import ChatAware
from ollama import ChatResponse, Message, Image, Client as RawClient
from pydantic import BaseModel




class ImagePrompt(BaseModel):
    content: str
    paths: list[Path] = []

    def as_message(self, message: Message) -> Message:
        message.content = self.content
        message.images = [Image(value=_) for _ in self.paths]
        return message


Prompt = str | ImagePrompt


class Client(ChatAware):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = RawClient()
        self.messages: list[dict | Message] = []

    def chat(self, *user_prompt: Prompt, **kwargs):
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=self.messages_from(*user_prompt),
        )
        output = response.message
        self.messages.append(output)
        return output.content

    def messages_from(self, *user_prompt: Prompt):

        for _ in user_prompt:
            message = Message(
                role='user',
            )
            match _:
                case str():
                    message.content = _
                case ImagePrompt():
                    message = _.as_message(message)

            self.messages.append(message)
        return self.messages
