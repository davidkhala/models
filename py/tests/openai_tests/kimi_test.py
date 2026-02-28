import json
import os
import unittest
from pathlib import Path
from unittest import skipIf, skipUnless

from openai import BadRequestError

from davidkhala.llm.openai.kimi import Global


class GlobalBase(unittest.TestCase):
    def setUp(self):
        api_key = os.getenv("API_KEY")
        self.client = Global(api_key)


class KimiTest(GlobalBase):

    def tearDown(self):
        """Clean up any files created during testing."""
        try:
            self.client.nuke()
        except Exception:
            pass

    def upload_one(self):
        self.client.upload(Path(__file__).parent / "fixtures" / "bitcoin.pdf")

    def test_file_upload(self):
        """Test uploading a file to Moonshot AI."""

        with self.assertRaises(BadRequestError) as e:
            self.client.upload(Path(__file__).parent / "fixtures" / "empty.pdf")
        self.assertEqual(e.exception.message,
                         "Error code: 400 - {'error': {'message': 'text extract error: 没有解析出内容', 'type': 'invalid_request_error'}}")

    @skipIf(os.environ.get('CI'), "requires API key and file upload")
    def test_files_list(self):
        """Test listing all uploaded files."""
        self.upload_one()
        files = self.client.files_list()
        print(files)
        self.assertIsInstance(files, list)

    def test_message(self):
        message = self.client.message_of(Path(__file__).parent / "fixtures" / "bitcoin.pdf")

        j = json.loads(message['content'])
        self.assertTrue(j['content'])
        self.assertEqual('application/pdf', j['file_type'])
        self.assertEqual('bitcoin.pdf', j['filename'])
        self.assertFalse(j['title'])
        self.assertEqual('file', j['type'])

    @skipIf(os.environ.get('CI'), "paid model")
    def test_chat_context(self):
        """Test that chat context is maintained across messages."""
        self.client.reset()
        self.client.as_chat(model="moonshot-v1-8k")

        r1 = self.client.chat("My name is Alice")
        r2 = self.client.chat("What is my name?")
        r3 = self.client.chat("How many questions have I asked?")

        # Second response should contain the name
        self.assertIn("Alice", r2[0])
        # Third response should indicate two questions
        self.assertTrue("two" in r3[0].lower() or "2" in r3[0])


    def test_list_models(self):
        """Test listing available models."""
        models = self.client.models
        print(models)
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

        has_kimi_model = any("moonshot" in mid or "kimi" in mid for mid in models)
        self.assertTrue(has_kimi_model, "Expected to find moonshot or kimi models")


if __name__ == "__main__":
    unittest.main()
