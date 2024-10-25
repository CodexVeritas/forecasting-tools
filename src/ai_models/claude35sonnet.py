from typing import Final
from src.ai_models.model_archetypes.anthropic_text_model import AnthropicTextToTextModel
from src.ai_models.ai_utils.ai_misc import clean_indents  # Keep this import here for easier imports into other files so prompts can keep proper indentation levels in code # NOSONAR

class Claude35Sonnet(AnthropicTextToTextModel):
    # See Anthropic Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: Final[str] = "claude-3-5-sonnet-20240620"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 50
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 40000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60

