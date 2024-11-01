import os
from typing import Literal
from unittest.mock import Mock

import pytest

from forecasting_tools.forecasting.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.team_manager import TeamManager
from forecasting_tools.util import file_manipulation
from tests.no_cost_expect_all_to_succeed.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)


async def test_collects_reports_on_open_questions(mocker: Mock) -> None:
    ForecastingTestManager.mock_forecaster_team_run_forecast(mocker)
    tournament_id = (
        ForecastingTestManager.TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN
    )
    manager = TeamManager()
    reports = await manager.run_forecasts_on_all_open_questions(tournament_id)
    questions_that_should_be_being_forecast_on = (
        MetaculusApi.get_all_questions_from_tournament(
            tournament_id, filter_by_open=True
        )
    )
    assert len(reports) == len(
        questions_that_should_be_being_forecast_on
    ), "Not all questions were forecasted on"


async def test_file_is_made_for_benchmark(mocker: Mock) -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    ForecastingTestManager.mock_forecaster_team_run_forecast(mocker)
    manager = TeamManager()

    file_path_to_save_reports = "logs/forecasts/benchmarks/"
    absolute_path = file_manipulation.get_absolute_path(
        file_path_to_save_reports
    )

    files_before = len(
        [
            f
            for f in os.listdir(absolute_path)
            if os.path.isfile(os.path.join(absolute_path, f))
        ]
    )

    await manager.benchmark_forecast_team("shallow")

    files_after = len(
        [
            f
            for f in os.listdir(absolute_path)
            if os.path.isfile(os.path.join(absolute_path, f))
        ]
    )
    assert (
        files_after > files_before
    ), "No new benchmark report file was created"


async def test_each_benchmark_mode_calls_forecaster_more_time(
    mocker: Mock,
) -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    manager = TeamManager()
    mock_run_forecast = (
        ForecastingTestManager.mock_forecaster_team_run_forecast(mocker)
    )
    modes: list[Literal["shallow", "medium", "deep"]] = [
        "shallow",
        "medium",
        "deep",
    ]
    num_calls_for_modes = []
    for mode in modes:
        score = await manager.benchmark_forecast_team(mode)
        assert isinstance(score, float), "The score should be a float"

        previous_calls = num_calls_for_modes[-1] if num_calls_for_modes else 0
        current_calls = mock_run_forecast.call_count - previous_calls
        num_calls_for_modes.append(current_calls)

        assert (
            current_calls > previous_calls
        ), "No new forecast calls were made"
