from __future__ import annotations

import asyncio
import os
import sys
import time

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

from forecasting_tools.forecasting.forecast_bots.team_manager import (
    TeamManager,
)
from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.util.custom_logger import CustomLogger


def run_morning_forecasts() -> None:
    CustomLogger.setup_logging()
    forecaster = TeamManager(time_to_wait_between_questions=65)
    TOURNAMENT_ID = MetaculusApi.AI_COMPETITION_ID_Q4
    try:
        reports = asyncio.run(
            forecaster.run_and_publish_forecasts_on_all_open_questions(
                TOURNAMENT_ID
            )
        )
    except Exception:
        reports = asyncio.run(
            forecaster.run_and_publish_forecasts_on_all_open_questions(
                TOURNAMENT_ID
            )
        )
    for report in reports:
        assert isinstance(report, BinaryReport)
        ForecastDatabaseManager.add_forecast_report_to_database(
            report, ForecastRunType.REGULAR_FORECAST
        )
        time.sleep(10)


if __name__ == "__main__":
    run_morning_forecasts()
