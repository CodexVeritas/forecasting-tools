from datetime import datetime

from forecasting_tools.forecasting.forecast_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.forecast_reports.metaculus_question import (
    DateQuestion,
)


class DateReport(ForecastReport):
    question: DateQuestion
    prediction: list[tuple[datetime, float]]
