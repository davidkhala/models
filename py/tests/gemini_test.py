import os
import unittest

from davidkhala.llm.gemini import Client, from_api_key


class ChatTest(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("API_KEY")

        self.client = Client(from_api_key(api_key, project="gcp-data-davidkhala"))

    def test_sample(self):
        self.client.as_chat('gemini-3.8-flash')
        self.client.n = 2
        r = self.client.chat('How does AI work?')
        print(r)
        self.assertEqual(2, len(r))
