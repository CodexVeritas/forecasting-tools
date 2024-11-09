import logging

import pytest

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_reports.metaculus_question import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.forecast_reports.report_organizer import (
    ReportOrganizer,
)
from forecasting_tools.forecasting.forecast_team.forecast_team import (
    ForecastTeam,
)

logger = logging.getLogger(__name__)

# Takes in Numeric, Date, and Binary
# Outputs corresponding report
# Corresponding reports have all fields
# Publishing forecast does not error
# Base rate questions created make sense
# Done

# TeamManager can take in a question and return a corresponding report


@pytest.mark.parametrize(
    "question_type", ReportOrganizer.get_all_question_types()
)
async def test_predicts_test_question(
    question_type: type[MetaculusQuestion],
) -> None:
    question = ReportOrganizer.get_live_example_question_of_type(question_type)
    assert isinstance(question, question_type)
    target_cost_in_usd = 2
    with MonetaryCostManager() as cost_manager:
        report = await ForecastTeam(question).run_forecast()
        logger.info(f"Report: \n{report}")
        logger.info(f"Cost of forecast: {cost_manager.current_usage}")
        logger.info(f"Report Explanation: \n{report.explanation}")
        expected_report_type = (
            ReportOrganizer.get_report_type_for_question_type(question_type)
        )
        assert isinstance(report, expected_report_type)
        assert cost_manager.current_usage <= target_cost_in_usd
        assert len(report.report_sections) > 1
        # TODO: Try to publish the report by picking questions that are non-competitive
