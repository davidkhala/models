from pathlib import Path
from typing import Literal

from openai import OpenAI
from openai.types import FileObject

from davidkhala.llm.model.chat import MessageDict
from davidkhala.llm.model.file import FileAware
from davidkhala.llm.openai import Client as BaseClient
from davidkhala.utils.syntax.format import mime_of


class Client(BaseClient, FileAware):

    def __init__(self, api_key: str, base_url: str):
        super().__init__(OpenAI(api_key=api_key, base_url=base_url))

    def upload(self, file: Path, purpose: Literal["file-extract", "image", "video"] | None = None) -> FileObject:
        """
        Supported file formats:
        .pdf, .txt, .csv, .doc, .docx, .xls, .xlsx, .ppt, .pptx, .md, .jpeg, .png, .bmp, .gif, .svg, .svgz, .webp, .ico, .xbm, .dib, .pjp, .tif, .pjpeg, .avif, .dot, .apng, .epub, .tiff, .jfif, .html, .json, .mobi, .log, .go, .h, .c, .cpp, .cxx, .cc, .cs, .java, .js, .css, .jsp, .php, .py, .py3, .asp, .yaml, .yml, .ini, .conf, .ts, .tsx,
        """
        if not purpose:
            match mime_of(file):
                case "application/pdf":
                    purpose = 'file-extract'
        return self.client.files.create(file=file, purpose=purpose)

    def file_get(self, file_id: str) -> FileObject:
        return self.client.files.retrieve(file_id)

    def files_list(self):
        return self.client.files.list().data

    def download(self, file_id: str) -> str:

        return self.client.files.content(file_id).text

    def files_delete(self, file_id: str):
        self.client.files.delete(file_id=file_id)

    def message_of(self, file: Path) -> MessageDict:
        file_id = self.upload(file).id
        return {
            'role': 'system',
            'content': self.download(file_id)
        }

    def nuke(self):
        files = self.files_list()
        for file in files:
            self.files_delete(file.id)


class Global(Client):
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.moonshot.ai/v1")
