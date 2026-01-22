import unittest

from davidkhala.models.vllm.run import server

class RunTest(unittest.IsolatedAsyncioTestCase):
    async def test_run(self):
        await server("deepseek-ai/DeepSeek-OCR")


class EnvTest(unittest.TestCase):
    def test_env(self):
        import torch
        self.assertFalse(torch.cuda.is_available()) # False for CPU

