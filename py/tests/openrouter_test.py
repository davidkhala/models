import os
import unittest
from unittest import skipIf

from davidkhala.llm.openrouter import Client as OpenRouterClient


class AdminTestCase(unittest.TestCase):
    from davidkhala.llm.openrouter import Admin
    def setUp(self):
        admin_key = os.environ.get('PROVISIONING_API_KEY')
        self.admin = AdminTestCase.Admin(admin_key)

    def test_keys(self):
        print(self.admin.keys)


class AppTestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get('API_KEY')
        self.openrouter = OpenRouterClient(api_key)


class InfoTestCase(AppTestCase):
    def test_connect(self):
        self.assertTrue(self.openrouter.connect())

    def test_models(self):
        r = self.openrouter.models
        print(r)
        for m in self.openrouter.list_models('embeddings'):
            print(m.id)


class ChatTestCase(AppTestCase):
    def test_free(self):
        self.openrouter.as_chat('openrouter/free', sys_prompt='You are a shiny girl today')
        r = self.openrouter.chat('Hello!')
        print(r)

    @skipIf(os.environ.get('CI'), "paid model")
    def test_ibm(self):
        self.openrouter.as_chat("ibm-granite/granite-4.0-h-micro")
        r = self.openrouter.chat("What is the meaning of life?")
        print(r)


class EmbedTestCase(AppTestCase):
    @skipIf(os.environ.get('CI'), "paid model")
    def test_embed(self):
        self.openrouter.as_embeddings('mistralai/codestral-embed-2505')

        r = self.openrouter.encode("apple", "banana")
        self.assertEqual(2, len(r))
        self.assertEqual(1536, len(r[0]))


if __name__ == "__main__":
    unittest.main()
