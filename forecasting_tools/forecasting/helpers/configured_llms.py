from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.ai_models.gpto1 import GptO1


class BasicLlm(Gpt4o):
    # NOTE: If need be, you can force an API key here through OpenAI Client class variable
    pass


class AdvancedLlm(GptO1):
    pass
