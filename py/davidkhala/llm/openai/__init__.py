from typing import Literal

from openai import OpenAI, AuthenticationError, PermissionDeniedError, HttpxBinaryResponseContent
from openai.types import Model

from davidkhala.llm.model import Connectable
from davidkhala.llm.model.embed import EmbeddingAware
from davidkhala.llm.model.garden import GardenAlike
from davidkhala.llm.model.videos import VideoAware


class Client(EmbeddingAware, VideoAware, GardenAlike, Connectable):

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

    def as_chat(self, model: str | None, sys_prompt: str = None):...

    def encode(self, *_input: str) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(_input),
            encoding_format=self.encoding_format
        )
        return [item.embedding for item in response.data]

    def video(self, prompt: str, seconds: Literal["4", "8", "12"] = 4, **kwargs) -> HttpxBinaryResponseContent:
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
