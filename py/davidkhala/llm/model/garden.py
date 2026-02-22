from abc import ABC, abstractmethod

from davidkhala.utils.protocol import ID


class GardenAlike(ABC):
    @abstractmethod
    def list_models(self) -> list[ID]: ...

    @property
    def models(self) -> list[str]:
        return [m.id for m in self.list_models()]

    @property
    @abstractmethod
    def free_models(self) -> list[str]: ...
