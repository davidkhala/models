class ModelAware:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: str | None = None


class Connectable:
    def connect(self) -> bool: ...

    def close(self): ...

    def __enter__(self):
        assert self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
