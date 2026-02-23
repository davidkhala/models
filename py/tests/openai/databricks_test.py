import unittest

from davidkhala.llm.openai.databricks import Client
from openai import NotFoundError


class HotmailFreeEdition(unittest.TestCase):
    o = 7474644422835710  # hotmail.com
    token = 'dapi' + '036089e12de676f44e7f5b1510fdd5e1'
    client = Client(o, token)

    def test_models(self):
        with self.assertRaises(NotFoundError) as e:
            _ = self.client.models
        self.assertEqual(e.exception.status_code, 404)
        self.assertEqual(
            "Error code: 404 - {'error_code': 'ENDPOINT_NOT_FOUND', 'message': 'Path must be of form /serving-endpoints/<endpoint_name>/invocations or /serving-endpoints/<endpoint_name>/served-models/<served_model_name>/invocations'}",
            e.exception.message)

    def test_chat(self):
        self.client.as_chat("databricks-gpt-oss-120b")
        responses = self.client.chat("What is an LLM agent?")

        for r in responses:
            print(r)


if __name__ == '__main__':
    unittest.main()
