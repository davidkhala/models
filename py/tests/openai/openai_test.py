import os
from unittest import skipIf, TestCase

from davidkhala.llm.openai.native import NativeClient


@skipIf(not os.environ.get('CI'), 'unsupported_country_region_territory')
class CITestCase(TestCase):
    def setUp(self):
        self.api_key = os.getenv("API_KEY")
        self.client = NativeClient(api_key=self.api_key)
    def test_connect(self):
        self.assertTrue(self.client.connect())

        
        
