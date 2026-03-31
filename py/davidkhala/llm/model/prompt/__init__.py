from pydantic import BaseModel
from typing_extensions import Literal


class TextContentPart(BaseModel):
    type: Literal['text'] = 'text'
    text: str
