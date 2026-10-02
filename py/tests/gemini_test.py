import os
import unittest

from davidkhala.llm.gemini import Client


class ChatTest(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("API_KEY") # Selected APIs: Agent Platform API

        self.client = Client(api_key, project="gcp-data-davidkhala")

    def test_sample(self):
        self.client.as_chat('gemini-3.8-flash')
        self.n = 2
        r = self.client.chat('How does AI work?')
        print(r)
