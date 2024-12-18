from __future__ import annotations

import asyncio
import logging

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.helpers.benchmarker import Benchmarker
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot() -> None:
    with MonetaryCostManager() as cost_manager:
        bot = MainBot()
        score = await Benchmarker().benchmark_forecast_bot(
            number_of_questions_to_test=100,
            forecast_bot=bot,
        )
        logger.info(f"Total Cost: {cost_manager.current_usage}")
        logger.info(f"Final Score: {score}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(benchmark_forecast_bot())
