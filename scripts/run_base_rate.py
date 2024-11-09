from __future__ import annotations

import asyncio

import dotenv

from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.sub_question_researchers.base_rate_researcher import (
    BaseRateReport,
    BaseRateResearcher,
)
from forecasting_tools.util.custom_logger import CustomLogger


def run_base_rate() -> None:
    question = input("Enter you question: ")
    base_rate_responder = BaseRateResearcher(question)
    report = asyncio.run(base_rate_responder.make_base_rate_report())
    escaped_question = "".join(
        char if char.isalnum() else "_" for char in question
    )
    log_path = f"logs/forecasts/base_rates/{escaped_question}.json"
    BaseRateReport.save_object_list_to_file_path([report], log_path)
    ForecastDatabaseManager.add_base_rate_report_to_database(
        report, ForecastRunType.REGULAR_BASE_RATE
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    run_base_rate()
