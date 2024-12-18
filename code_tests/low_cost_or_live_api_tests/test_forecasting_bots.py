import logging
from unittest.mock import Mock

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_bots.bot_lists import (
    get_cheap_bot_question_type_pairs,
)
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.questions import (
    Question,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "question_type, bot", get_cheap_bot_question_type_pairs()
)
async def test_predicts_test_question(
    question_type: type[Question],
    bot: ForecastBot,
) -> None:

    question = ReportOrganizer.get_live_example_question_of_type(question_type)
    assert isinstance(question, question_type)
    target_cost_in_usd = 2
    with MonetaryCostManager() as cost_manager:
        report = await bot.forecast_question(question)
        logger.info(f"Cost of forecast: {cost_manager.current_usage}")
        logger.info(f"Report Explanation: \n{report.explanation}")
        expected_report_type = (
            ReportOrganizer.get_report_type_for_question_type(question_type)
        )
    assert isinstance(report, expected_report_type)
    assert cost_manager.current_usage <= target_cost_in_usd
    assert len(report.report_sections) > 1
    assert report.prediction is not None
    assert report.explanation is not None
    assert report.price_estimate is not None
    assert report.minutes_taken is not None
    assert report.question is not None

    await report.publish_report_to_metaculus()


async def test_collects_reports_on_open_questions(mocker: Mock) -> None:
    bot_type = TemplateBot
    bot = bot_type()
    ForecastingTestManager.mock_forecast_bot_forecast(bot_type, mocker)
    tournament_id = (
        ForecastingTestManager.TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN
    )
    reports = await bot.forecast_on_tournament(tournament_id)
    questions_that_should_be_being_forecast_on = (
        MetaculusApi.get_all_open_questions_from_tournament(tournament_id)
    )
    assert len(reports) == len(
        questions_that_should_be_being_forecast_on
    ), "Not all questions were forecasted on"
