import os
import unittest
from unittest import skipIf


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ.get("API_KEY")


class ChatTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        from davidkhala.llm.zai.chat import GlobalClient
        self.client = GlobalClient(self.api_key, glm_coding_plan=True)

    def test_chat(self):
        self.client.as_chat('glm-4.7')
        r = self.client.chat('hello')
        print(r)


class VideoTestCase(BaseTestCase):
    def setUp(self):
        super().setUp()
        from davidkhala.llm.zai.video import GlobalClient
        self.client = GlobalClient(self.api_key)
        self.client.model = "cogvideox-3"

    @skipIf(os.environ.get("CI"), "Insufficient balance")
    def test_video(self):
        r = self.client.video("A cat sitting on floor")



if __name__ == "__main__":
    unittest.main()
