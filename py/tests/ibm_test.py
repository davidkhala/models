import os
import unittest

from davidkhala.llm.ibm import Client


class Watsonx(unittest.TestCase):
    def setUp(self):
        project = os.getenv("PROJECT") or 'ed4ce828-ed18-46a4-be4a-ca44dae7860d'
        region = 'ca-tor'
        api_key = os.getenv("API_KEY")
        self.client = Client(project, region=region, api_key=api_key)

    def test_chat(self):
        self.client.reset()
        self.client.as_chat('mistralai/mistral-small-3-1-24b-instruct-2503')
        r = self.client.chat('How far is Paris from Bangalore?')
        print(self.client.on_response(r))

    def test_choices_chat(self):
        self.client.reset()
        self.client.n = 2
        self.client.as_chat('mistralai/mistral-small-3-1-24b-instruct-2503')
        r = self.client.chat('How far is Paris from Bangalore?')

        choices = self.client.on_response(r)
        self.assertEqual(len(choices), self.client.n)
        self.assertTrue('The distance between Paris, France, and Bangalore, India, can vary' in choices[0]) # seed check


if __name__ == '__main__':
    unittest.main()
