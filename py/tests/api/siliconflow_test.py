import os
import time
import unittest

from davidkhala.llm.api.siliconflow import SiliconFlow
from requests import HTTPError

class BaseTest(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get('API_KEY')
        self._ = SiliconFlow(api_key)

class CommonTests(BaseTest):
    def test_models(self):
        _models = self._.list_models()
        print(_models)


class ChatTestCase(BaseTest):

    def test_chat(self):
        self._.as_chat('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
        r = self._.chat('who am I?')
        print(r)


class EmbeddingTestCase(BaseTest):
    def test_array(self):
        self._.as_embeddings('BAAI/bge-m3')
        start_time = time.time()
        r = self._.encode("abc--------------------------------", "edf-----------------")
        print(time.time() - start_time)
        self.assertEqual(2, len(r))

    def test_empty(self):
        self._.as_embeddings('BAAI/bge-m3')
        with self.assertRaises(HTTPError) as e:
            self._.encode("")
        self.assertEqual(400, e.exception.response.status_code)


class RerankTestCase(BaseTest):

    def test_model_compare(self):
        self._.model = 'BAAI/bge-reranker-v2-m3'
        query = 'apple'
        docs = ["apple", "banana", "fruit", "vegetable"]
        self.assertEqual('apple', self._.which(query, docs)[0])
        self._.model = 'Qwen/Qwen3-Reranker-8B'  # unnatural model, and inconsistent
        self.assertIn(self._.which(query, docs)[0], ['apple', 'banana'])
        self._.model = 'Qwen/Qwen3-Reranker-4B' # unnatural model, and inconsistent
        self.assertIn(self._.which(query, docs)[0], ['banana', 'fruit'])
        self._.model = 'Qwen/Qwen3-Reranker-0.6B'
        self.assertEqual('apple', self._.which(query, docs)[0])

    def test_null(self):
        self._.model = 'BAAI/bge-reranker-v2-m3'
        with self.assertRaises(HTTPError) as e:
            self._.which('apple', [])  # bad request
        self.assertEqual(400, e.exception.response.status_code)


if __name__ == '__main__':
    unittest.main()
