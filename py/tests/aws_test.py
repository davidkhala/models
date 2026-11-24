import os
import unittest

from botocore.exceptions import ClientError

from davidkhala.llm.aws import Client


class QuickStart(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_region_limit(self):
        self.client.as_chat("anthropic.claude-haiku-4-5-20251001-v1:0")
        if os.environ.get("CI"):
            # CI runners are in a supported region, so the call should succeed
            self.client.chat("Hi")
            return
        with self.assertRaisesRegex(
                ClientError,
                r"ValidationException.*Converse.*Access to Anthropic models is not allowed from unsupported countries, regions, or territories"
        ):
            self.client.chat("Hi")

    def test_kimi(self):
        self.client.as_chat("moonshotai.kimi-k3.1")
        r = self.client.chat("Write a one-sentence bedtime story about a unicorn.")
        print(r)

if __name__ == '__main__':
    unittest.main()
