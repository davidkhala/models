from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Image(BaseModel):
    image_url: str
    detail: Literal['auto', 'low', 'high'] = 'auto'  # Image detail level for vision models

    def expand(self):
        from davidkhala.llm.model.prompt.image import URL, ContentPart

        return ContentPart(image_url=URL(url=self.image_url, detail=self.detail))


class File(BaseModel):
    url: str | None = None
    path: Path | None = None

    def expand(self):
        from davidkhala.llm.model.prompt.file import File, ContentPart
        from davidkhala.utils.syntax.url import filename_from
        from davidkhala.utils.syntax.format import mime_of
        from davidkhala.utils.syntax.format import Base64
        if self.url:
            return ContentPart(file=File(filename=filename_from(self.url), file_data=self.url))
        if self.path:
            return ContentPart(file=File(
                filename=self.path.name,
                file_data=f"data:{mime_of(self.path)};base64,{Base64.encode_file(self.path)}")
            )
        assert False


Prompt = str | Image | File
