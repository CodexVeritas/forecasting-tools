from forecasting_tools.forecasting.forecast_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.forecast_reports.metaculus_question import (
    MultipleChoiceQuestion,
)


class MultipleChoiceReport(ForecastReport):
    question: MultipleChoiceQuestion
    prediction: list[tuple[str, float]]
