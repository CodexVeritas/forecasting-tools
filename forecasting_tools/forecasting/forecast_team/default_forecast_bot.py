from forecasting_tools.forecasting.forecast_team.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOptionSet,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
)


class DefaultForecastBot(ForecastBot):

    def __init__(
        self,
        *args,
        number_of_background_questions_to_ask: int = 3,
        number_of_base_rate_questions_to_ask: int = 3,
        number_of_base_rates_to_do_deep_research_on: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.number_of_background_questions_to_ask = (
            number_of_background_questions_to_ask
        )
        self.number_of_base_rate_questions_to_ask = (
            number_of_base_rate_questions_to_ask
        )
        self.number_of_base_rates_to_do_deep_research_on = (
            number_of_base_rates_to_do_deep_research_on
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        raise NotImplementedError(
            "DefaultForecastBot does not implement run_research"
        )

    async def run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        raise NotImplementedError(
            "DefaultForecastBot does not implement run_forecast_on_binary"
        )

    async def run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionSet]:
        raise NotImplementedError(
            "DefaultForecastBot does not implement run_forecast_on_multiple_choice"
        )

    async def run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError(
            "DefaultForecastBot does not implement run_forecast_on_numeric"
        )

    def _extract_forecast_from_binary_rationale(self, rationale: str) -> float:
        raise NotImplementedError(
            "DefaultForecastBot does not implement _extract_forecast_from_binary_rationale"
        )

    def _extract_forecast_from_multiple_choice_rationale(
        self, rationale: str
    ) -> PredictedOptionSet:
        raise NotImplementedError(
            "DefaultForecastBot does not implement _extract_forecast_from_multiple_choice_rationale"
        )

    def _extract_forecast_from_numeric_rationale(
        self, rationale: str
    ) -> NumericDistribution:
        raise NotImplementedError(
            "DefaultForecastBot does not implement _extract_forecast_from_numeric_rationale"
        )
