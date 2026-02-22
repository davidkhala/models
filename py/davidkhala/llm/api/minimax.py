from pathlib import Path
from typing import Literal

import requests
from davidkhala.utils.http_request import default_on_response
from requests import Response, HTTPError

from davidkhala.llm.api import API
from davidkhala.llm.model.file import FileAware

Purpose = Literal['voice_clone', 'prompt_audio', 't2a_async_input']
"""
See in https://platform.minimax.io/docs/api-reference/file-management-upload#body-purpose
    - voice_clone: For quick cloning of the original voice file (supports mp3, m4a, wav formats).
    - prompt_audio: Sample audio for voice cloning (supports mp3, m4a, wav formats).
    - t2a_async_input: Text file in the request body for asynchronous long-text-to-speech synthesis.(text-to-speech)
"""


class Minimax(API, FileAware):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)

        def on_response(r: requests.Response) -> dict | None:
            r_dict = default_on_response(r)
            if r_dict is None:
                return None
            status = r_dict['base_resp']['status_code']
            msg = r_dict['base_resp']['status_msg']
            if status:
                derived = Response()
                derived.url = r.url
                derived.status_code = status
                derived.reason = msg
                derived.raise_for_status()
                raise HTTPError(f"{derived.status_code} Client Error: {derived.reason} for url: {derived.url}",
                                response=derived)
            else:
                assert status == 0
                assert msg == "success"
            return r_dict

        self.on_response = on_response

    def files_list(self):
        url = f"{self.base_url}/files/list"
        r = self.request(url, 'GET')
        return r['files']

    def file_get(self, file_id: int) -> dict | None:
        url = f"{self.base_url}/files/retrieve"
        try:
            r = self.request(url, 'GET', params={
                'file_id': file_id
            })
            return r['file']
        except HTTPError as err:
            if err.response.status_code == 2013 and err.response.reason == 'invalid params, file not found':
                return None
            raise err

    def files_delete(self, file_id: int, purpose: Purpose) -> int:
        url = f"{self.base_url}/files/delete"
        r = self.request(url, 'POST', json={
            "file_id": file_id,
            "purpose": purpose
        })

        return r['file_id']

    def nuke(self):
        r = []
        for _ in self.files_list():
            file_id = _['file_id']
            r.append(file_id)
            self.files_delete(_['file_id'], _['purpose'])

    def upload(self, file: Path, purpose: Purpose):
        """
        :param file:
        :param purpose:
        """
        url = f"{self.base_url}/files/upload"

        with open(file, 'rb') as f:
            r = self.request(url, 'POST', data={
                'purpose': purpose
            }, files={
                'file': (file.name, f)
            })
        return r['file']

class CN(Minimax):
    def __init__(self, api_key: str):
        super().__init__(api_key, 'https://api.minimaxi.com')


class Global(Minimax):
    def __init__(self, api_key: str):
        super().__init__(api_key, 'https://api.minimax.io')
