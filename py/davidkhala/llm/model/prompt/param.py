from pathlib import Path
from typing import Literal

from davidkhala.utils.syntax.format import data_url_of
from pydantic import BaseModel, model_validator


class BytesAware(BaseModel):
    url: str | None = None
    path: Path | None = None

    @model_validator(mode='after')
    def _require_source(self):
        if not self.path and not (self.url and self.url.strip()):
            raise ValueError("either url or path is required")
        return self

    def source(self, timeout: int=30) -> bytes:
        from urllib.parse import urlparse
        from urllib.request import urlopen
        if self.path:

            if not self.path.is_file():
                raise FileNotFoundError(f"File not found: {self.path}")
            return self.path.read_bytes()

        if self.url and self.url.strip():
            parsed = urlparse(self.url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"Invalid URL: {self.url}")
            with urlopen(self.url, timeout=timeout) as resp:
                return resp.read()


class Image(BytesAware):
    detail: Literal['auto', 'low', 'high'] = 'auto'  # Image detail level for vision models

    @property
    def image_url(self) -> str:
        return self.url if self.url else data_url_of(self.path)


from davidkhala.utils.syntax.url import filename_from


class File(BytesAware):

    @property
    def filename(self) -> str:
        return filename_from(self.url) if self.url else self.path.name

    @property
    def file_data(self) -> str:
        return self.url if self.url else data_url_of(self.path)


Prompt = str | Image | File
