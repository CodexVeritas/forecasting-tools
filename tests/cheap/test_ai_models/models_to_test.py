from src.ai_models.basic_model_interfaces.ai_model import AiModel
from src.ai_models.basic_model_interfaces.incurs_cost import IncursCost
from src.ai_models.basic_model_interfaces.outputs_text import OutputsText
from src.ai_models.basic_model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from src.ai_models.basic_model_interfaces.retryable_model import RetryableModel
from src.ai_models.basic_model_interfaces.time_limited_model import TimeLimitedModel
from src.ai_models.basic_model_interfaces.token_limited_model import TokenLimitedModel
from src.ai_models.basic_model_interfaces.tokens_incur_cost import TokensIncurCost
from src.ai_models.claude35sonnet import Claude35Sonnet
from src.ai_models.exa_searcher import ExaSearcher
from src.ai_models.gpt4o import Gpt4o
from src.ai_models.gpt4ovision import Gpt4oVision
from src.ai_models.metaculus4o import Gpt4oMetaculusProxy
from src.ai_models.perplexity import Perplexity
from src.ai_models.gpto1 import GptO1


class ModelsToTest:
    ALL_MODELS = [
        Gpt4o,
        Gpt4oMetaculusProxy,
        Gpt4oVision,
        GptO1,
        Claude35Sonnet,
        Perplexity,
        ExaSearcher,
    ]
    BASIC_MODEL_LIST: list[type[AiModel]] = [
        model for model in ALL_MODELS if issubclass(model, AiModel)
    ]
    RETRYABLE_LIST: list[type[RetryableModel]] = [
        model for model in ALL_MODELS if issubclass(model, RetryableModel)
    ]
    TIME_LIMITED_LIST: list[type[TimeLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, TimeLimitedModel)
    ]
    REQUEST_LIMITED_LIST: list[type[RequestLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, RequestLimitedModel)
    ]
    TOKEN_LIMITED_LIST: list[type[TokenLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, TokenLimitedModel)
    ]
    INCURS_COST_LIST: list[type[IncursCost]] = [
        model for model in ALL_MODELS if issubclass(model, IncursCost)
    ]
    OUTPUTS_TEXT: list[type[OutputsText]] = [
        model for model in ALL_MODELS if issubclass(model, OutputsText)
    ]
    TOKENS_INCUR_COST_LIST: list[type[TokensIncurCost]] = [
        model for model in ALL_MODELS if issubclass(model, TokensIncurCost)
    ]
