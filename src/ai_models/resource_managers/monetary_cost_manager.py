from __future__ import annotations
from src.ai_models.resource_managers.hard_limit_manager import HardLimitManager
from src.ai_models.resource_managers.hard_limit_manager import HardLimitExceededError # For other files to easily import from this file #NOSONAR


class MonetaryCostManager(HardLimitManager):
    """
    This class is a subclass of HardLimitManager that is specifically for monetary costs.
    Assume every cost is in USD

    As of Aug 27 2024, the manager does not track predicted costs.
    For instance if you run 50 coroutines that cost 10c, and your limit is $1,
    all 50 will be let through (not 10).
    The cost will not register until the coroutines finish.
    """

    def __enter__(self) -> MonetaryCostManager:
        super().__enter__()
        return self