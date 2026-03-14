from zai import ZaiClient


class GlobalClient:
    def __init__(self, api_key, *, glm_coding_plan):
        super().__init__()
        self.client = ZaiClient(api_key=api_key, base_url='https://api.z.ai/api/coding/paas/v4')
        if not glm_coding_plan:
            self.client.base_url = self.client.default_base_url
