from abc import ABC, abstractmethod


class VideoAware(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size: str = "1280x720"  # Video resolution

    @abstractmethod
    def video(self, prompt: str, seconds: int, **kwargs): ...
