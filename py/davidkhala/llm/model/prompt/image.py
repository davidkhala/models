from typing import Literal

from pydantic import BaseModel
class URL(BaseModel):
    url: str
    detail: Literal['auto', 'low', 'high']


class ContentPart(BaseModel):
    type:Literal['image_url'] = 'image_url'
    image_url: URL