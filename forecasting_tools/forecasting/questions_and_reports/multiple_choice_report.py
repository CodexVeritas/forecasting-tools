from pydantic import BaseModel

from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    MultipleChoiceQuestion,
)


class PredictedOption(BaseModel):
    option: str
    probability: float


class PredictedOptionSet(BaseModel):
    predicted_options: list[PredictedOption]


class MultipleChoiceReport(ForecastReport):
    question: MultipleChoiceQuestion
    prediction: PredictedOptionSet
