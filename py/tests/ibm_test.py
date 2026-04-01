import os
import unittest
from davidkhala.llm.ibm import Client


class Watsonx(unittest.TestCase):
    def setUp(self):
        project = os.getenv("PROJECT")
        region = 'ca-tor'
        api_key = os.getenv("API_KEY")
        self.client = Client(project, region=region, api_key=api_key)

    def test_chat(self):
        self.client.as_chat('mistralai/mistral-small-3-1-24b-instruct-2503')
        r = self.client.chat('How far is Paris from Bangalore?')
        print(r)


if __name__ == '__main__':
    unittest.main()
