import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_team.research_coordinator import (
    ResearchCoordinator,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    BinaryQuestion,
    DateQuestion,
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
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)
from forecasting_tools.util import async_batching


class ResearchPredictionCollection(BaseModel):
    research_report: str
    summary_report: str
    predictions: list[ReasonedPrediction]


class ForecastBot(ABC):

    def __init__(
        self,
        *,
        research_reports_per_question: int = 3,
        predictions_per_research_report: int = 5,
        use_research_summary_to_forecast: bool = True,
    ) -> None:
        assert (
            research_reports_per_question > 0
        ), "Must run at least one research report"
        assert (
            predictions_per_research_report > 0
        ), "Must run at least one prediction"
        self.research_reports_per_question = research_reports_per_question
        self.predictions_per_research_report = predictions_per_research_report
        self.use_research_summary_to_forecast = (
            use_research_summary_to_forecast
        )

    async def forecast_on_tournament(
        self, tournament_id: int, publish: bool = True
    ) -> list[ForecastReport]:
        questions = MetaculusApi.get_all_open_questions_from_tournament(
            tournament_id
        )
        return await self.run_multiple_questions(questions, publish)

    async def run_multiple_questions(
        self, questions: list[MetaculusQuestion], publish: bool = False
    ) -> list[ForecastReport]:
        reports: list[ForecastReport] = []
        for question in questions:
            report = await self.run_question(question)
            reports.append(report)
        if publish:
            for report in reports:
                await report.publish_report_to_metaculus()
        return reports

    async def run_question(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        with MonetaryCostManager() as cost_manager:
            start_time = time.time()
            tasks = [
                self._create_research_prediction_collection(question)
                for _ in range(self.research_reports_per_question)
            ]
            research_prediction_collections, _ = (
                async_batching.run_coroutines_while_removing_and_logging_exceptions(
                    tasks
                )
            )
            report_type = ReportOrganizer.get_report_type_for_question_type(
                type(question)
            )
            all_predictions = [
                reasoned_prediction.prediction_value
                for research_prediction_collection in research_prediction_collections
                for reasoned_prediction in research_prediction_collection.predictions
            ]
            aggregated_prediction = await report_type.aggregate_predictions(
                all_predictions,
                question,
            )
            end_time = time.time()
            time_spent_in_minutes = (end_time - start_time) / 60
            final_cost = cost_manager.current_usage

        unified_explanation = self._create_unified_explanation(
            question,
            research_prediction_collections,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )

        return report_type(
            question=question,
            prediction=aggregated_prediction,
            explanation=unified_explanation,
            price_estimate=final_cost,
            minutes_taken=time_spent_in_minutes,
        )

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        research_coordinator = ResearchCoordinator(question)
        summary_report = (
            await research_coordinator.summarize_full_research_report(research)
        )
        return summary_report

    @abstractmethod
    async def run_research(self, question: MetaculusQuestion) -> str:
        pass

    @abstractmethod
    async def run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        pass

    @abstractmethod
    async def run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionSet]:
        pass

    @abstractmethod
    async def run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        pass

    async def _create_research_prediction_collection(
        self, question: MetaculusQuestion
    ) -> ResearchPredictionCollection:
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            research
            if self.use_research_summary_to_forecast
            else summary_report
        )

        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r: await self.run_forecast_on_binary(
                q, r
            )
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = (
                lambda q, r: await self.run_forecast_on_multiple_choice(q, r)
            )
        elif isinstance(question, NumericQuestion):
            forecast_function = (
                lambda q, r: await self.run_forecast_on_numeric(q, r)
            )
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        reasoned_predictions = [
            forecast_function(question, research_to_use)
            for _ in range(self.predictions_per_research_report)
        ]

        return ResearchPredictionCollection(
            research_report=research,
            summary_report=summary_report,
            predictions=reasoned_predictions,
        )

    def _create_unified_explanation(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchPredictionCollection],
        aggregated_prediction: Any,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        report_type = ReportOrganizer.get_report_type_for_question_type(
            type(question)
        )

        summaries = []
        full_research_reports = []
        rationales = []
        for i, collection in enumerate(research_prediction_collections):
            forecaster_prediction_bullet_points = ""
            for forecast in collection.predictions:
                readable_prediction = report_type.make_readable_prediction(
                    forecast.prediction_value
                )
                forecaster_prediction_bullet_points += (
                    f"- *Forecaster {i + 1}*: {readable_prediction}\n"
                )

            new_summary = clean_indents(
                f"""
                ## Report {i + 1} Summary
                ### Forecasts
                {forecaster_prediction_bullet_points}

                ### Research Summary
                {collection.summary_report}
                """
            )
            summaries.append(new_summary)

            modified_research_report = self.__add_report_number_to_headings(
                i + 1, collection.research_report
            )
            full_research_reports.append(modified_research_report)

            for j, forecast in enumerate(collection.predictions):
                new_rationale = clean_indents(
                    f"""
                    ## R{i + 1}: Forecaster {j + 1} Reasoning
                    {forecast.reasoning}
                    """
                )
                rationales.append(new_rationale)

        combined_summaries = "\n".join(summaries)
        combined_research_reports = "\n".join(full_research_reports)
        combined_rationales = "\n".join(rationales)
        full_explanation_without_summary = clean_indents(
            f"""
            # SUMMARY
            *Question*: {question.question_text}\n
            *Final Prediction*: {report_type.make_readable_prediction(aggregated_prediction)}\n
            *Total Cost*: ${round(final_cost,2)}
            *Time Spent*: {round(time_spent_in_minutes, 2)} minutes

            {combined_summaries}

            # RESEARCH
            {combined_research_reports}

            # FORECASTS
            {combined_rationales}
            """
        )
        return full_explanation_without_summary

    @classmethod
    def __add_report_number_to_headings(
        cls,
        report_number: int,
        markdown: str,
    ) -> str:
        lines = markdown.split("\n")
        modified_content = ""
        for line in lines:
            if line.startswith("## "):
                line = f"## R{report_number}: {line[3:]}"
            modified_content += line + "\n"
        return modified_content
