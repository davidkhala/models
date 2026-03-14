from zai import ZaiClient


class GlobalClient:
    def __init__(self, api_key, *, glm_coding_plan: bool):
        super().__init__()
        base_url = 'https://api.z.ai/api/coding/paas/v4' if glm_coding_plan else 'https://api.z.ai/api/paas/v4'
        self.client = ZaiClient(api_key=api_key, base_url=base_url)
        if not glm_coding_plan:
            assert base_url == self.client.default_base_url
