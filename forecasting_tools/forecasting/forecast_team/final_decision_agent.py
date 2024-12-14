import logging
from typing import Any

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_team.research_coordinator import (
    ResearchCoordinator,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


class FinalDecisionAgent:

    def __init__(
        self,
        research_as_markdown: str,
        question: MetaculusQuestion,
        number_of_predictions_to_run: int,
        cost_manager: MonetaryCostManager,
    ) -> None:
        assert (
            number_of_predictions_to_run > 0
        ), "Must run at least one prediction"
        assert research_as_markdown, "Research must be provided"
        self.research_as_markdown = research_as_markdown
        self.question = question
        self.report_type = ReportOrganizer.get_report_type_for_question_type(
            type(self.question)
        )
        self.number_of_predictions_to_run = number_of_predictions_to_run
        self.cost_manager = cost_manager
        self.__research_summary: str | None = None

    async def run_decision_agent(self) -> ForecastReport:
        try:
            research_summary = (
                await self.__get_research_summary_and_populate_if_empty()
            )
        except Exception as e:
            logger.error(f"Error in making research summary: {e}")
            research_summary = "Error in making research summary"

        logger.info(f"Running {self.number_of_predictions_to_run} predictions")
        final_prediction_coroutines = [
            self.report_type.run_prediction(self.question, research_summary)
            for _ in range(self.number_of_predictions_to_run)
        ]
        reasoned_predictions, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                final_prediction_coroutines
            )
        )
        if len(reasoned_predictions) == 0:
            raise ValueError("All forecasts errored")
        logger.info(
            f"{len(reasoned_predictions)} predictions successfully ran"
        )
        aggregated_prediction = await self.report_type.aggregate_predictions(
            [
                prediction.prediction_value
                for prediction in reasoned_predictions
            ]
        )
        explanation = await self.__create_unified_explanation(
            reasoned_predictions, aggregated_prediction
        )
        report = self.report_type(
            question=self.question,
            explanation=explanation,
            prediction=aggregated_prediction,
            price_estimate=self.cost_manager.current_usage,
        )
        logger.info("Compiled final report")
        return report

    async def __get_research_summary_and_populate_if_empty(self) -> str:
        if self.__research_summary:
            return self.__research_summary
        research_coordinator = ResearchCoordinator(self.question)
        cleaned_summary_markdown = (
            await research_coordinator.summarize_full_research_report(
                self.research_as_markdown
            )
        )
        self.__research_summary = cleaned_summary_markdown
        logger.info("Made research summary for final decision agent")
        return cleaned_summary_markdown

    async def __create_unified_explanation(
        self,
        reasoned_predictions: list[ReasonedPrediction],
        aggregated_prediction: Any,
    ) -> str:
        assert self.__research_summary

        forecaster_prediction_bullet_points = ""
        for i, forecast in enumerate(reasoned_predictions):
            readable_prediction = self.report_type.make_readable_prediction(
                forecast.prediction_value
            )
            forecaster_prediction_bullet_points += (
                f"- *Forecaster {i + 1}*: {readable_prediction}\n"
            )

        combined_reasoning = ""
        for i, forecast in enumerate(reasoned_predictions):
            combined_reasoning += f"## Reasoning from forecaster {i + 1}\n"
            combined_reasoning += forecast.reasoning
            combined_reasoning += "\n\n"

        full_explanation_without_summary = clean_indents(
            f"""
            # SUMMARY
            *Question*: {self.question.question_text}\n
            *Final Prediction*: {self.report_type.make_readable_prediction(aggregated_prediction)}\n
            *Total Cost*: ${round(self.cost_manager.current_usage, 2)}

            ## Forecaster Team Summary
            {forecaster_prediction_bullet_points}

            {self.__research_summary}

            # RESEARCH
            {self.research_as_markdown}

            # FORECASTS
            {combined_reasoning}
            """
        )
        return full_explanation_without_summary
