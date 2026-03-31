from pydantic import BaseModel
from typing_extensions import Literal

from davidkhala.llm.model.prompt import TextContentPart
from davidkhala.llm.model.prompt.image import ContentPart as ImageContentPart


class FileAnnotation(BaseModel):
    hash: str
    name: str | None
    content: list[TextContentPart | ImageContentPart]


class ContentPart(BaseModel):
    type: Literal['file'] = 'file'
    file: FileAnnotation
