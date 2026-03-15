from typing import List

from typing_extensions import Literal
from zai import ZaiClient
from zai.types.video import VideoResult

import davidkhala.llm.zai
from davidkhala.llm.model.videos import VideoAware


class Client(VideoAware):
    def __init__(self, *args, **kwargs):
        self.client: ZaiClient = None
        kwargs.pop('glm_coding_plan', None) # No coding plan has video generate capability
        super().__init__(*args, **kwargs)
        self.fps: Literal[30, 60] = 30  # Frame rate
        self.quality: Literal["quality", "speed"] = "speed"
        self.audio = False

    def video(self, prompt: str, seconds: Literal[5, 10] = 5, *, image_url: str = None) -> List[VideoResult]:
        response = self.client.videos.generations(
            model=self.model, prompt=prompt,
            image_url=image_url,
            quality=self.quality, with_audio=self.audio,
            duration=seconds,
            size=self.size,  # Video resolution, supports up to 4K (e.g., "3840x2160")
            fps=self.fps,
        )

        while True:
            result = self.client.videos.retrieve_videos_result(id=response.id)
            match result.task_status:
                case 'SUCCESS':
                    return result.video_result
                case 'FAIL':
                    raise RuntimeError(f'Video {result.id} failed with request id={result.request_id}')


class GlobalClient(davidkhala.llm.zai.GlobalClient, Client):
    ...
