from src.ai_models.model_archetypes.perplexity_text_model import PerplexityTextModel
from typing import Final
from src.ai_models.ai_utils.ai_misc import clean_indents  # Keep this import here for easier imports into other files so prompts can keep proper indentation levels in code # NOSONAR



class Perplexity(PerplexityTextModel):
    MODEL_NAME: Final[str] = "llama-3.1-sonar-huge-128k-online"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 16 # Technically 20, but has problems at 20 for some reason
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 120
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 2000000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
    PRICE_PER_TOKEN: Final[float] = 0.000005
    PRICE_PER_REQUEST: Final[float] = 0.005

