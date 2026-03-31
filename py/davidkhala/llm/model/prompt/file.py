from pydantic import BaseModel
from typing_extensions import Literal


class File(BaseModel):
    filename: str
    file_data: str


class ContentPart(BaseModel):
    type: Literal["file"] = "file"
    file: File
