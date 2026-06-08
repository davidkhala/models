
from davidkhala.llm.model.chat import ChatAware
from ollama import chat, ChatResponse

class Client(ChatAware):
    def chat(self, *user_prompt, **kwargs):
        response: ChatResponse = chat(
            model = self.model,
            message = self.messages_from(*user_prompt),
        )