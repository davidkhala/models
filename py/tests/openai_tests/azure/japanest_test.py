import os
import unittest

from davidkhala.llm.openai.azure import OpenAIClient

project = 'ai-japanest'


class OpenAITestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("API_KEY")

        self.client = OpenAIClient(api_key, project)

    def test_connect(self):
        self.assertTrue(self.client.connect())
        print(self.client.models)

    def test_embedding(self):
        self.client.as_embeddings("text-embedding-3-large")
        print(self.client.encode("Attention is all you need"))


if __name__ == '__main__':
    unittest.main()
