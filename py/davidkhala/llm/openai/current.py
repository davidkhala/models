from openai import OpenAI
from openai.resources.responses import Responses
from openai.types.responses import ResponseInputImageParam, Response
from openai.types.responses.response_input_item_param import Message

from davidkhala.llm.model.chat import ChatAware
from davidkhala.llm.model.prompt.param import Prompt, Image, File
from davidkhala.llm.openai import Client as BaseClient


class Client(BaseClient, ChatAware):
    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.instructions: str | None = None
        self.messages: list = []

    def as_chat(self, model: str | None, sys_prompt: str = None):
        self.model = model
        if sys_prompt is not None:
            self.instructions = sys_prompt

    def chat(self, *user_prompt:Prompt, **kwargs)->Response:
        responses: Responses = self.client.responses
        response:Response = responses.create(
            model=self.model,
            input=self.messages_from(*user_prompt),
            instructions=self.instructions,
            **kwargs
        )
        self.messages.extend(response.output)

        return response

    def messages_from(self, *user_prompt: Prompt) -> list[Message]:

        from openai.types.responses import ResponseInputContentParam, ResponseInputTextParam, ResponseInputFileParam
        content: list[ResponseInputContentParam] = []
        for _ in user_prompt:
            match _:
                case str():
                    content.append(ResponseInputTextParam(text=_, type='input_text'))
                case Image():

                    content.append(ResponseInputImageParam(detail=_.detail, image_url=_.image_url, type='input_image'))
                case File():
                    if _.url:
                        p = ResponseInputFileParam(file_url=_.url, type='input_file')
                    else:
                        p = ResponseInputFileParam(file_data=_.file_data, filename=_.filename, type='input_file')
                    content.append(p)

        message = Message(role='user', content=content)

        self.messages.append(message)
        return self.messages
