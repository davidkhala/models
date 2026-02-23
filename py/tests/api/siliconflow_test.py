import os
import unittest
from unittest import skipIf

from requests import HTTPError

from davidkhala.llm.api.siliconflow import SiliconFlow


class BaseTest(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get('API_KEY') or 'sk-uqwpfglnfcvuynsnmdxefowawwzoaqqyjzsztbwgdlstmkdx'
        self._ = SiliconFlow(api_key)


class ModelsTestCase(BaseTest):
    def test_models(self):
        _models = self._.list_models()
        print(_models)
    def test_hack_free_models(self):
        # https://cloud.siliconflow.cn/biz-server/api/v1/playground/17885302824/biz_info
        model_id = "17885302824"
        # TODO WIP
        ...


class ChatTestCase(BaseTest):
    from davidkhala.llm.model.chat import ImagePrompt
    def test_free(self):
        self._.reset()
        self._.as_chat('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
        r = self._.chat('who am I?')
        self.assertEqual(1, len(r))
        print(r)

    @skipIf(os.environ.get('CI'), "paid model")
    def test_n(self):
        self._.reset()
        self._.as_chat('Pro/MiniMaxAI/MiniMax-M2.5')
        self._.n = 2
        r = self._.chat('who am I?')
        self.assertEqual(2, len(r))
        print(r)

    @skipIf(os.environ.get('CI'), "paid model")
    def test_n_multimodal(self):
        self._.reset()
        self._.as_chat('Qwen/Qwen3-VL-32B-Instruct')
        self._.n = 2
        r1 = self._.chat(ChatTestCase.ImagePrompt(
            text='Are these 2 images identical?',
            image_url=[
                "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png",
                "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png"
            ],
        ))
        print(r1)


class EmbeddingTestCase(BaseTest):
    def test_array(self):
        self._.as_embeddings('BAAI/bge-m3')
        r = self._.encode("abc--------------------------------", "edf-----------------")
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
        self._.model = 'Qwen/Qwen3-Reranker-4B'  # unnatural model, and inconsistent
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
