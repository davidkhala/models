import os
import unittest
from davidkhala.llm.ollama.cloud import Client as CLoudClient
from davidkhala.llm.ollama import Client


class LocalTest(unittest.TestCase):
    def setUp(self):
        self._ = Client()
    def test_chat(self):
        os.system('ollama pull mistral')
        self._.as_chat('mistral')
        r = self._.chat('What is 17 × 23?')
        print(r)


class CloudTest(unittest.TestCase):
    def setUp(self):
        api_key = os.getenv("API_KEY")
        self.web = CLoudClient(api_key)


class WebTest(CloudTest):
    def test_search(self):
        r = self.web.search("What is Ollama?")
        for _ in r:
            print(_)

    def test_fetch(self):
        r = self.web.fetch('https://ollama.com')
        print(r)
