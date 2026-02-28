import os
import unittest
from unittest import skipIf


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("API_KEY")

from davidkhala.llm.openai.minimax import Global
class GlobalTest(BaseTest):
    def setUp(self):
        super().setUp()
        self._ = Global(self.api_key)
class OpenAITest(GlobalTest):

    @skipIf(os.environ.get('CI'), "paid model")
    def test_chat(self):
        self._.reset()
        self._.as_chat("MiniMax-M2.5")
        r = self._.chat("Hi, how are you?")
        print(r[0])
        r1 = self._.chat("Who are you?")
        print(r1[0])
        r2 = self._.chat("how many questions I have asked you in the conversation?")
        self.assertTrue('three' in r2[0] or '3' in r2[0])

    @skipIf(os.environ.get('CI'), "paid model")
    def test_reasoning(self):
        self._.reset()
        self._.as_chat("MiniMax-M2.5")
        r = self._.chat("Hi, how are you?")
        for reason_dict in r[1]:
            self.assertIsInstance(reason_dict, dict)
            print(reason_dict)


class AnthropicTest(GlobalTest):

    @skipIf(os.environ.get('CI'), "paid model")
    def test_chat(self):
        self._.reset()
        self._.as_chat("MiniMax-M2.5")
        r = self._.chat("Hi, how are you?")
        print(r)
        r1 = self._.chat("Who are you?")
        print(r1)


if __name__ == "__main__":
    unittest.main()
