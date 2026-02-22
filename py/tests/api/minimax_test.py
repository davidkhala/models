import os
import unittest
from pathlib import Path


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("API_KEY")


class FSTest(BaseTest):
    def setUp(self):
        super().setUp()
        from davidkhala.llm.api.minimax import Global
        self.api = Global(self.api_key)

    def test_file_list(self):
        r = self.api.files_list()
        print(r)

    def test_file_get(self):
        noexist = 369079598493921
        r1 = self.api.file_get(noexist)
        self.assertIsNone(noexist)

    def test_file_upload(self):
        file = Path(__file__).parent / "fixtures" / "empty.txt"
        r = self.api.upload(file, 't2a_async_input')
        print(r)
        r1 = self.api.file_get(r['file_id'])
        self.assertDictEqual(r1, r)

    def tearDown(self):
        self.api.nuke()


if __name__ == "__main__":
    unittest.main()
