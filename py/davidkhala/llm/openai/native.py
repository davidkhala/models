from typing import Literal

from openai import OpenAI, HttpxBinaryResponseContent

from davidkhala.llm.model.videos import VideoWare
from davidkhala.llm.openai import Client


class NativeClient(Client, VideoWare):
    def __init__(self, api_key, base_url=None):
        super().__init__(OpenAI(
            api_key=api_key,
            base_url=base_url
        ))

    def chat(self, *user_prompt, web_search: Literal["low", "medium", "high"] = None, **kwargs):
        opts = {
            **kwargs
        }
        if web_search:
            from openai.types.chat.completion_create_params import WebSearchOptions
            opts['web_search_options'] = WebSearchOptions(search_context_size=web_search)
        return super().chat(*user_prompt, **opts)

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
