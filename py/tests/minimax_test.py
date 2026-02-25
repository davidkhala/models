import os
import unittest
from unittest import skipIf


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv("API_KEY")


class OpenAITest(BaseTest):
    from davidkhala.llm.openai.minimax import Global
    def setUp(self):
        super().setUp()
        self._ = OpenAITest.Global(self.api_key)

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


class AnthropicTest(BaseTest):
    from davidkhala.llm.anthropic.minimax import Global
    def setUp(self):
        super().setUp()
        self._ = AnthropicTest.Global(self.api_key)

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
