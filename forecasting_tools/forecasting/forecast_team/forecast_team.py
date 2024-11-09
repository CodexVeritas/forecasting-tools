from __future__ import annotations

import datetime
import logging
import time

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_team.final_decision_agent import (
    FinalDecisionAgent,
)
from forecasting_tools.forecasting.forecast_team.research_coordinator import (
    ResearchCoordinator,
)
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_question import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    NumericQuestion,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


class ForecastTeam:

    def __init__(
        self,
        question: MetaculusQuestion,
        number_of_reports_to_aggregate: int = 3,
        number_of_predictions_per_report: int = 3,
        number_of_background_questions_to_ask: int = 3,
        number_of_base_rate_questions_to_ask: int = 3,
        number_of_base_rates_to_do_deep_research_on: int = 0,
    ) -> None:
        working_question_types = [
            BinaryQuestion,
            NumericQuestion,
            DateQuestion,
        ]
        assert any(
            isinstance(question, t) for t in working_question_types
        ), f"Question must be one of the following types: {working_question_types}"
        self.question = question
        self.num_reports_to_aggregate = number_of_reports_to_aggregate
        self.num_predictions_per_report = number_of_predictions_per_report
        self.number_of_background_questions_to_ask = (
            number_of_background_questions_to_ask
        )
        self.number_of_base_rate_questions_to_ask = (
            number_of_base_rate_questions_to_ask
        )
        self.number_of_base_rates_to_do_deep_research_on = (
            number_of_base_rates_to_do_deep_research_on
        )

    async def run_forecast(self) -> ForecastReport:
        start_time = time.time()
        with MonetaryCostManager() as cost_manager:
            report_tasks = [
                self.__research_and_make_forecast()
                for _ in range(self.num_reports_to_aggregate)
            ]
            log_function = lambda error, _: logger.exception(
                f"Error while researching and making forecast in ForecastTeam: {error.__class__.__name__} Exception - {error}"
            )
            reports, _ = (
                async_batching.run_coroutines_while_removing_and_logging_exceptions(
                    report_tasks, action_on_exception=log_function
                )
            )
            assert len(reports) > 0, "No reports were created"
            end_time = time.time()
            duration_in_minutes = (end_time - start_time) / 60

            report_type = type(reports[0])
            combined_report = await report_type.combine_report_list_into_one(
                reports, cost_manager.current_usage, duration_in_minutes
            )
            report_type.save_object_list_to_file_path(
                [combined_report], self.__create_log_file_path()
            )
            return combined_report

    async def __research_and_make_forecast(self) -> ForecastReport:
        with MonetaryCostManager() as cost_manager:
            research_manager = ResearchCoordinator(self.question)
            combined_markdown = (
                await research_manager.create_full_markdown_research_report(
                    self.number_of_background_questions_to_ask,
                    self.number_of_base_rate_questions_to_ask,
                    self.number_of_base_rates_to_do_deep_research_on,
                )
            )
            decision_agent = FinalDecisionAgent(
                combined_markdown,
                self.question,
                self.num_predictions_per_report,
                cost_manager,
            )
            report = await decision_agent.run_decision_agent()
            return report

    def __create_log_file_path(self) -> str:
        shortened_question_text = (
            self.question.question_text.replace(" ", "_")
            .replace("?", "")
            .replace(":", "")[:20]
        )
        now_as_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return f"logs/forecasts/forecast_team/{now_as_string}-{shortened_question_text}-{self.question.question_id}.json"
