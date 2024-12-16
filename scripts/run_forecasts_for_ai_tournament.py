from __future__ import annotations

import asyncio
import os
import sys

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.custom_logger import CustomLogger


async def run_morning_forecasts() -> None:
    CustomLogger.setup_logging()
    forecaster = MainBot(
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="logs/forecasts/forecast_bot/",
        skip_previously_forecasted_questions=True,
    )
    TOURNAMENT_ID = MetaculusApi.AI_COMPETITION_ID_Q4
    reports = await forecaster.forecast_on_tournament(TOURNAMENT_ID)
    for report in reports:
        await asyncio.sleep(5)
        ForecastDatabaseManager.add_forecast_report_to_database(
            report, ForecastRunType.REGULAR_FORECAST
        )


if __name__ == "__main__":
    asyncio.run(run_morning_forecasts())
