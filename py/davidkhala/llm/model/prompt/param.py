from pathlib import Path
from typing import Literal

from davidkhala.utils.syntax.format import data_url_of
from pydantic import BaseModel

class Image(BaseModel):
    url: str | None = None
    path: Path | None = None
    detail: Literal['auto', 'low', 'high'] = 'auto'  # Image detail level for vision models

    @property
    def image_url(self) -> str:
        return self.url if self.url else data_url_of(self.path)


from davidkhala.utils.syntax.url import filename_from


class File(BaseModel):
    url: str | None = None
    path: Path | None = None

    @property
    def filename(self) -> str:
        return filename_from(self.url) if self.url else self.path.name

    @property
    def file_data(self) -> str:
        return self.url if self.url else data_url_of(self.path)


Prompt = str | Image | File
