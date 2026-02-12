import os
import unittest
from davidkhala.llm.atlas import Client


class VoyageTest(unittest.TestCase):
    def test_sample(self):
        api_key = os.environ.get("API_KEY")
        c = Client(api_key)
        c.as_embeddings()
        result = c.encode("MongoDB is redefining what a database is in the AI era.")
        self.assertEqual(1024, len(result[0]))


if __name__ == '__main__':
    unittest.main()
