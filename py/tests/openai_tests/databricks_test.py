import unittest

from openai import NotFoundError, BadRequestError

from davidkhala.llm.openai.databricks import Client


class HotmailFreeEdition(unittest.TestCase):
    o = 7474644422835710  # hotmail.com
    token = 'dapi' + '036089e12de676f44e7f5b1510fdd5e1'
    client = Client(o, token)

    def test_models(self):
        with self.assertRaises(BadRequestError) as e:
            _ = self.client.models
        self.assertEqual(e.exception.status_code, 400)
        self.assertEqual(
            "Error code: 400 - Request path '/mlflow/v1/models' doesn't match any known API type and is classified as an unmanaged api request. Set the Databricks-Model-Provider-Service header to the name of the model provider service to forward this request to.",
            e.exception.message)

    def test_chat(self):
        self.client.as_chat("system.ai.gpt-oss-120b")
        response, reason = self.client.chat("What is an LLM agent?")

        for r in response:
            print(r)

    def test_qwen(self):
        self.client.as_chat("system.ai.qwen3-next-80b-a3b-instruct")
        responses, reason = self.client.chat("What is an LLM agent?")

        for r in responses:
            print(r)


if __name__ == '__main__':
    unittest.main()
