from davidkhala.utils.protocol import ID


class GardenAlike:
    def list_models(self) -> list[ID]: ...

    @property
    def models(self) -> list[str]:
        return [m.id for m in self.list_models()]

    @property
    def free_models(self) -> list[str]:
        ...
        return []
