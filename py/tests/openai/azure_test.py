import os
import unittest

from davidkhala.llm.openai.azure import ModelDeploymentClient, OpenAIClient


class OpenAITestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("API_KEY")
        project = os.environ.get("PROJECT")
        self.client = OpenAIClient(api_key, project)

    def test_connect(self):
        self.assertTrue(self.client.connect())

    def test_chat(self):
        self.client.as_chat()
        print(self.client.chat("hello"))


class ModelDeploymentTestCase(unittest.TestCase):
    def setUp(self):
        key = os.environ.get("DEPLOYMENT_KEY")
        deployment = os.environ.get("DEPLOYMENT")
        self.client = ModelDeploymentClient(key, deployment)

    def test_connect(self):
        self.assertTrue(self.client.connect())

    def test_chat(self):
        self.client.as_chat("gpt-4o", "You are a helpful assistant.")
        response = self.client.chat("Don't reply me anything now.", "What is your model name?")
        self.assertEqual(1, len(response))
        print(response[0])

    def test_embedding(self):
        self.client.as_embeddings("text-embedding-3-large")
        print(self.client.encode("Attention is all you need"))

    def test_ocr(self):
        from pathlib import Path
        file = Path(__file__).parent.parent / "fixtures" / "transcript.png"
        from davidkhala.llm.openai.azure import FieldProperties
        schema = {
            'Student': FieldProperties(required=True),
            'Date of Birth': FieldProperties(required=True),
            'Weighted GPA': FieldProperties(required=True),
            'Gender': FieldProperties(required=True),
            'Credits Earned': FieldProperties(),
        }
        self.client.as_chat("gpt-4o",
                            "You are a professional OCR parser. Produce the output strictly according to the JSON schema")
        r = self.client.process(file, schema)
        print(r)


if __name__ == '__main__':
    unittest.main()
