from typing import Protocol, Any


class SecretProtocol(Protocol):
    api_key: str


class ModelProtocol(Protocol):
    model: str | None


class SDKProtocol(Protocol):
    client: Any
