import os
import unittest
from davidkhala.llm.zai.chat import Client
class BaseTestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("API_KEY")
        self.client = Client(api_key)

class ChatTestCase(BaseTestCase):
    def test_chat(self):
        # FIXME zai.core._errors.APIReachLimitError: Error code: 429, with error text {"error":{"code":"1113","message":"Insufficient balance or no resource package. Please recharge."}}
        self.client.as_chat('glm-4.7')
        r = self.client.chat('hello')
        print(r)

if __name__ == "__main__":
    unittest.main()