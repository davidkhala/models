from pathlib import Path
from typing import Literal

from davidkhala.utils.syntax.format import Base64, mime_of
from pydantic import BaseModel


def data_url_of(path: Path):
    """
    :return: base64 encoded file in a data URL.
    TODO: import from davidkhala.utils.syntax later
    """
    return f"data:{mime_of(path)};base64,{Base64.encode_file(path)}"


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
