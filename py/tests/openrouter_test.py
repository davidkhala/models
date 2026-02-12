import os
import unittest
from unittest import skipIf

from davidkhala.llm.openrouter import Client as OpenRouterClient, Admin


class SDKTestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get('API_KEY')
        admin_key = os.environ.get('PROVISIONING_API_KEY')
        self.openrouter = OpenRouterClient(api_key)
        self.admin = Admin(admin_key)

    def test_connect(self):
        self.assertTrue(self.openrouter.connect())

    def test_keys(self):
        print(self.admin.keys)

    def test_models(self):
        r = self.openrouter.models
        print(r)

    def test_chat(self):
        self.openrouter.as_chat('openrouter/free', sys_prompt='You are a shiny girl today')
        r = self.openrouter.chat('Hello!')
        print(r)

    @skipIf(os.environ.get('CI'), "paid model")
    def test_ibm(self):
        self.openrouter.as_chat("ibm-granite/granite-4.0-h-micro")
        r = self.openrouter.chat("What is the meaning of life?")
        print(r)


if __name__ == "__main__":
    unittest.main()
