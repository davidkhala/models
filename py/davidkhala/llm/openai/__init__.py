from typing import Literal

from openai import OpenAI, AuthenticationError, PermissionDeniedError, HttpxBinaryResponseContent
from openai.types import Model
from openai.types.chat import ChatCompletion

from davidkhala.llm.model import Connectable
from davidkhala.llm.model.chat import on_response, ChoicesChat, DeterministicChat
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.garden import GardenAlike
from davidkhala.llm.model.videos import VideoAware


class Client(ChoicesChat, EmbeddingAware, VideoAware, DeterministicChat, GardenAlike, Connectable):

    def __init__(self, client: OpenAI):
        super().__init__()
        self.client: OpenAI = client
        self.encoding_format: Literal["float", "base64"] = "float"

    def connect(self):
        try:
            self.list_models()
            return True
        except AuthenticationError | PermissionDeniedError:
            return False

    def list_models(self) -> list[Model]:
        return self.client.models.list().data

    def encode(self, *_input: str) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(_input),
            encoding_format=self.encoding_format
        )
        return [item.embedding for item in response.data]

    def chat_create(self, *user_prompt, **kwargs) -> ChatCompletion:
        if hasattr(self, 'seed'):
            kwargs['seed'] = self.seed
        # TODO **Starting a new project?** We recommend trying
        #         [Responses](https://platform.openai.com/docs/api-reference/responses) to take
        #         advantage of the latest OpenAI platform features. Compare
        #         [Chat Completions with Responses](https://platform.openai.com/docs/guides/responses-vs-chat-completions?api-mode=responses).
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages_from(*user_prompt),
            n=self.n,
            **kwargs
        )

    def chat(self, *user_prompt, **kwargs):
        response: ChatCompletion = self.chat_create(*user_prompt, **kwargs)
        return on_response(response, self.n)

    def video(self, prompt: str, seconds: Literal["4", "8", "12"] = 4) -> HttpxBinaryResponseContent:
        from openai.types import Video
        from openai.resources import Videos
        videos: Videos = self.client.videos
        video: Video = videos.create_and_poll(
            model=self.model,  # e.g. "sora-2",
            prompt=prompt,
            size=self.size,
            seconds=seconds
        )

        if video.error:
            raise video.error

        return videos.download_content(video.id)

    def close(self):
        self.client.close()
