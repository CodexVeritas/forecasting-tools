from __future__ import annotations

import asyncio
import logging
import os
import time

import dotenv

from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def republish_reports_for_today() -> None:

    date_str: str = time.strftime("%Y-%m-%d")
    forecast_path = "logs/forecasts/forecast_team"
    absolute_forecast_path = file_manipulation.get_absolute_path(forecast_path)

    # Get all files in the directory
    files = os.listdir(absolute_forecast_path)

    # Filter files for the specified date
    target_files = [f for f in files if f.startswith(date_str)]

    if not target_files:
        logger.warning(f"No forecast files found for date {date_str}")
        return

    logger.info(f"Found {len(target_files)} forecast files for {date_str}")

    all_reports: list[BinaryReport] = []
    for file_name in target_files:
        file_path = os.path.join(absolute_forecast_path, file_name)
        reports = BinaryReport.convert_project_file_path_to_object_list(
            file_path
        )
        all_reports.extend(reports)

    for report in all_reports:
        assert isinstance(report, BinaryReport)
        try:
            await report.publish_report_to_metaculus()
            logger.info(
                f"Successfully published forecast for question {report.question.post_id}"
            )
            ForecastDatabaseManager.add_forecast_report_to_database(
                report, ForecastRunType.REGULAR_FORECAST
            )
            time.sleep(10)  # Wait between publications to avoid rate limiting
        except Exception as e:
            logger.error(f"Failed to publish report: {e}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    asyncio.run(republish_reports_for_today())
