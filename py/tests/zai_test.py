import os
import unittest
from davidkhala.llm.zai.chat import Client
class BaseTestCase(unittest.TestCase):
    def setUp(self):
        api_key = os.environ.get("API_KEY")
        self.client = Client(api_key, glm_coding_plan=False)

class ChatTestCase(BaseTestCase):
    def test_chat(self):
        self.client.as_chat('glm-4.7')
        r = self.client.chat('hello')
        print(r)

if __name__ == "__main__":
    unittest.main()