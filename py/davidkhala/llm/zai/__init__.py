from zai import ZaiClient

class Client:
    def __init__(self, api_key):
        super().__init__()
        self.client = ZaiClient(api_key=api_key)


