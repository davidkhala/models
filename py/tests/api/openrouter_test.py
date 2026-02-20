import os
import unittest

from davidkhala.utils.syntax.path import resolve
from requests import HTTPError

from davidkhala.llm.api.openrouter import OpenRouter as OpenRouterAPI
from davidkhala.llm.model.chat import FilePrompt


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get('API_KEY')
        self.openrouter = OpenRouterAPI(api_key)


class ChatTestCase(BaseTestCase):

    def test_chat(self):
        self.openrouter.as_chat("openrouter/free")
        r = self.openrouter.chat('who am I?')
        print(r)

    def test_url_pdf(self):
        self.openrouter.as_chat('openrouter/free')
        r = self.openrouter.chat(FilePrompt(
            text='Summerize this pdf',
            url=['https://bitcoin.org/bitcoin.pdf']
        ))
        print(r)

    def test_local_pdf(self):
        self.openrouter.as_chat('openrouter/free')
        r1 = self.openrouter.chat(FilePrompt(
            text='Summerize this pdf',
            url=['https://bitcoin.org/bitcoin.pdf'],
            path=[resolve(__file__, '../../fixtures/empty.pdf')]
        ))
        print(self.openrouter.messages)
        r2 = self.openrouter.chat('Learn from the annotations')
        print(r2)
    def test_mix_pdf(self):
        self.openrouter.as_chat('openrouter/free')
        r1 = self.openrouter.chat(FilePrompt(
            text='Summerize these 2 pdf',
            url=['https://bitcoin.org/bitcoin.pdf'],
            path=[resolve(__file__, '../../fixtures/empty.pdf')]
        ))
        print(r1)

    def test_chat_models(self):
        self.openrouter.as_chat("deepseek/deepseek-r1-0528:free", "deepseek/deepseek-chat-v3.1")
        r = self.openrouter.chat('who am I?')
        print(r)


class ModelsTestCase(BaseTestCase):
    def test_models(self):
        models = self.openrouter.free_models
        self.assertGreaterEqual(len(models), 26)
        print(models)


class EmbeddingsTestCase(BaseTestCase):
    def test_embed(self):
        self.openrouter.as_embeddings('mistralai/codestral-embed-2505')

        r = self.openrouter.encode("apple", "banana")
        self.assertEqual(2, len(r))
        self.assertEqual(1536, len(r[0]))


class RegionLimitTestCase(BaseTestCase):
    def test_google_limit(self):
        if os.environ.get('CI'):
            self.skipTest("Gemini is available in GitHub runner region")
        for model in ['google/gemma-3n-e2b-it:free']:
            self.openrouter.as_chat(model)
            with self.assertRaises(HTTPError) as e:
                self.openrouter.chat('-')
            self.assertEqual(e.exception.response.status_code, 400)

    def test_google(self):
        self.openrouter.as_chat('google/gemini-2.5-flash-lite-preview-09-2025')
        r = self.openrouter.chat('-')
        print(r)

    def test_openai(self):
        allowed_models = ['openai/gpt-oss-20b:free', 'openai/gpt-5-nano']
        for model in allowed_models:
            self.openrouter.as_chat(model)
            r = self.openrouter.chat('return True')
            print(model, r)

    def test_openai_limit(self):
        if os.environ.get('CI'):
            self.skipTest("openai is available in GitHub runner region")
        self.openrouter.as_chat('openai/gpt-4.1-nano')
        with self.assertRaises(HTTPError) as e:
            self.openrouter.chat('-')
        self.assertEqual(e.exception.response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
