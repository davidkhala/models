import os
import unittest
from unittest import skipIf

from davidkhala.utils.syntax.path import resolve

from davidkhala.llm.openrouter.admin import Admin
from davidkhala.llm.openrouter.client import Client as OpenRouterClient


class AdminTestCase(unittest.TestCase):

    def setUp(self):
        admin_key = os.environ.get('PROVISIONING_API_KEY')
        self.admin = Admin(admin_key)

    def test_keys(self):
        print(list(self.admin.keys))


class FreeTestCase(unittest.TestCase):
    def setUp(self):
        api_key = 'sk-or-v1' + '-e7b1d56e3237f21aab0de23eaa58885ea09adde9a8dd31f18cabfb90ca4ea17b'
        self.openrouter = OpenRouterClient(api_key)


class PaidTestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get('API_KEY')
        self.openrouter = OpenRouterClient(api_key)


class InfoTestCase(FreeTestCase):
    def test_connect(self):
        self.assertTrue(self.openrouter.connect())

    def test_models(self):
        r = self.openrouter.free_models
        print(len(r), r)
        for m in self.openrouter.list_models('embeddings'):
            print(m.id)


from davidkhala.llm.model.prompt.param import File as FilePrompt


class FreeChatTestCase(FreeTestCase):
    def test_free(self):
        self.openrouter.as_chat('openrouter/free', sys_prompt='You are a shiny girl today')
        r = self.openrouter.chat('Hello!')
        print(r)

    def test_url_pdf(self):
        self.openrouter.as_chat('openrouter/free')
        self.openrouter.seed = None
        r = self.openrouter.chat('Summerize this pdf', FilePrompt(url='https://bitcoin.org/bitcoin.pdf'))
        print(r)

    def test_local_pdf(self):
        self.openrouter.as_chat('openrouter/free')
        self.openrouter.seed = None
        r = self.openrouter.chat('Summerize this pdf', FilePrompt(
            path=resolve(__file__, '../fixtures/bitcoin.pdf')
        ))
        print(r)


@skipIf(os.environ.get('CI'), "paid model")
class PaidChatTestCase(PaidTestCase):
    def test_seed(self):
        self.openrouter.as_chat("ibm-granite/granite-4.0-h-micro")
        r = self.openrouter.chat('Hello!')
        print(r)
        self.assertEqual(r, "Hello! How can I assist you today?")

    def test_embed(self):
        self.openrouter.as_embeddings('mistralai/codestral-embed-2505')
        r = self.openrouter.encode("apple", "banana")
        self.assertEqual(2, len(r))
        self.assertEqual(1536, len(r[0]))

    def test_ibm(self):
        self.openrouter.as_chat("ibm-granite/granite-4.0-h-micro")
        r = self.openrouter.chat("What is the meaning of life?")
        print(r)


if __name__ == "__main__":
    unittest.main()
