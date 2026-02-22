import os
import unittest
from unittest import skipIf


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("API_KEY")


class OpenAITest(BaseTest):
    ...


class AnthropicTest(BaseTest):
    from davidkhala.llm.anthropic.minimax import Global
    def setUp(self):
        super().setUp()
        self._ = AnthropicTest.Global(api_key=self.api_key)

    @skipIf(os.environ.get('CI'), "paid model")
    def test_chat(self):
        self._.reset()
        self._.as_chat("MiniMax-M2.5")
        r = self._.chat("Hi, how are you?")
        print(r)
        r1 = self._.chat("Who are you?")
        print(r1)

    @skipIf(os.environ.get('CI'), "paid model")
    def test_n_chat(self):
        self._.reset()
        self._.as_chat("MiniMax-M2.5")
        self._.n = 2
        r = self._.chat("Hi, how are you?")
        print(r)


if __name__ == "__main__":
    unittest.main()
