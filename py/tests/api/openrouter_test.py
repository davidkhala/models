import os
import unittest
from unittest import skipIf

from davidkhala.utils.syntax.path import resolve
from requests import HTTPError


class BaseTestCase(unittest.TestCase):
    api_key: str

    def setUp(self):
        from davidkhala.llm.api.openrouter import OpenRouter as OpenRouterAPI
        self.openrouter = OpenRouterAPI(self.api_key)


@skipIf(os.environ.get('CI'), "paid api-key")
class PaidTestCase(BaseTestCase):
    def setUp(self):
        self.api_key = os.environ.get('API_KEY')
        super().setUp()

from davidkhala.llm.model.prompt.param import File as FilePrompt
class FreeTestCase(BaseTestCase):
    def setUp(self):
        self.api_key = 'sk-or-v1' + '-e7b1d56e3237f21aab0de23eaa58885ea09adde9a8dd31f18cabfb90ca4ea17b'
        super().setUp()
        self.openrouter.as_chat("openrouter/free")

    def test_chat(self):
        r = self.openrouter.chat('who am I?')
        print(r)

    def test_url_pdf(self):

        self.openrouter.seed = None
        r = self.openrouter.chat('Summerize this pdf', FilePrompt(
            url='https://bitcoin.org/bitcoin.pdf'
        ))
        print(r)

    def test_local_pdf(self):
        r = self.openrouter.chat(
            'Read this pdf',
            FilePrompt(path=resolve(__file__, '../../fixtures/bitcoin.pdf'))
        )
        print(r)
    def test_models(self):
        models = self.openrouter.free_models
        self.assertGreaterEqual(len(models), 11)
        print(len(models), models)

        # 'nvidia/nemotron-3.5-super-vl:free', 'nvidia/nemotron-3.5-lightning:free', 'nvidia/nemotron-3-ultra-550b-a55b:free', 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
        # 'google/gemma-4-26b-a4b-it:free', 'google/gemma-4-31b-it:free'

class ChatTestCase(PaidTestCase):

    def test_chat_models(self):
        self.openrouter.as_chat("deepseek/deepseek-r1-0528:free", "deepseek/deepseek-chat-v3.1")
        r = self.openrouter.chat('who am I?')
        print(r)


class EmbeddingsTestCase(PaidTestCase):
    def test_embed(self):
        self.openrouter.as_embeddings('mistralai/codestral-embed-2505')

        r = self.openrouter.encode("apple", "banana")
        self.assertEqual(2, len(r))
        self.assertEqual(1536, len(r[0]))


class RegionLimitTestCase(PaidTestCase):
    @skipIf(os.environ.get('CI'), "gemini is available in GitHub runner region")
    def test_google_limit(self):
        for model in ['google/gemma-4-31b-it:free', 'google/gemma-4-26b-a4b-it:free']:
            self.openrouter.as_chat(model)
            with self.assertRaises(HTTPError) as e:
                self.openrouter.chat('-')
            self.assertEqual(e.exception.response.status_code, 400)

    @skipIf(not os.environ.get('CI'), "gemini is available in GitHub runner region")
    def test_google(self):
        self.openrouter.as_chat('google/gemini-3.8-flash')
        r = self.openrouter.chat('-')
        print(r)

    @skipIf(not os.environ.get('CI'), "openai is available in GitHub runner region")
    def test_openai(self):
        allowed_models = ['openai/gpt-oss-20b', 'openai/gpt-5-nano']
        self.openrouter.seed = None
        for model in allowed_models:
            self.openrouter.as_chat(model)
            r = self.openrouter.chat('-')
            print(model, r)

    @skipIf(os.environ.get('CI'), "openai is available in GitHub runner region")
    def test_openai_limit(self):
        self.openrouter.as_chat('openai/gpt-4.1-nano')
        self.openrouter.seed = None
        with self.assertRaises(HTTPError) as e:
            self.openrouter.chat('-')
        self.assertEqual(e.exception.response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
