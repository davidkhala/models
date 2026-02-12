from davidkhala.llm.model.chat import CompareChatAware
from davidkhala.llm.model.garden import GardenAlike


class OpenRouterModel(CompareChatAware, GardenAlike):
    def __init__(self):
        super().__init__()
        del self.n  # openrouter has no `n` parameter support TODO Is it global? or limited to openrouter?

    @property
    def free_models(self) -> list[str]:
        l = list(
            map(lambda model: model['id'],
                filter(lambda model: model['id'].endswith(':free'), self.list_models())
                )
        )
        l.append('openrouter/free')
        return l
