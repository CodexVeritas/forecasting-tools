import numpy as np
import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
)


def create_test_binary_question(
    community_prediction: float | None = 0.7,
) -> BinaryQuestion:
    question = ForecastingTestManager.get_question_safe_to_pull_and_push_to()
    question.community_prediction_at_access_time = community_prediction
    return question


def create_test_binary_report(
    prediction: float = 0.5, community_prediction: float | None = 0.7
) -> BinaryReport:
    return BinaryReport(
        question=create_test_binary_question(community_prediction),
        prediction=prediction,
        explanation="# Test Report\nThis is a test report",
    )


def test_prediction_validation() -> None:
    # Valid predictions
    report = create_test_binary_report(prediction=0.5)
    assert report.prediction == pytest.approx(0.5)

    report = create_test_binary_report(prediction=0.0001)
    assert report.prediction == pytest.approx(0.0001)

    report = create_test_binary_report(prediction=0.9999)
    assert report.prediction == pytest.approx(0.9999)

    # Invalid predictions
    with pytest.raises(ValueError):
        create_test_binary_report(prediction=-0.1)

    with pytest.raises(ValueError):
        create_test_binary_report(prediction=1.1)

    with pytest.raises(ValueError):
        create_test_binary_report(prediction=2.0)


async def test_aggregate_predictions() -> None:
    question = create_test_binary_question()
    predictions = [0.1, 0.2, 0.3, 0.4, 0.5]

    result = await BinaryReport.aggregate_predictions(predictions, question)
    assert result == pytest.approx(0.3)  # Median of predictions

    # Test invalid predictions
    with pytest.raises(Exception):
        await BinaryReport.aggregate_predictions([-0.1, 0.5], question)

    with pytest.raises(Exception):
        await BinaryReport.aggregate_predictions([1.1, 0.5], question)


def test_inversed_expected_log_score() -> None:
    # Test with valid community prediction
    report = create_test_binary_report(
        prediction=0.6, community_prediction=0.7
    )
    score = report.inversed_expected_log_score
    assert score is not None
    expected_score = -1 * (0.7 * np.log2(0.6) + 0.3 * np.log2(0.4))
    assert score == pytest.approx(expected_score)
    assert score > 0

    # Test better prediction less than worse prediction
    better_report = create_test_binary_report(
        prediction=0.6, community_prediction=0.7
    )
    worse_report = create_test_binary_report(
        prediction=0.4, community_prediction=0.7
    )
    better_score = better_report.inversed_expected_log_score
    worse_score = worse_report.inversed_expected_log_score
    assert better_score is not None
    assert worse_score is not None
    assert better_score < worse_score

    # Test with None community prediction
    report = create_test_binary_report(
        prediction=0.6, community_prediction=None
    )
    assert report.inversed_expected_log_score is None


def test_deviation_points() -> None:
    # Test with valid community prediction
    report = create_test_binary_report(
        prediction=0.6, community_prediction=0.7
    )
    deviation = report.deviation_points
    assert deviation is not None
    assert deviation == pytest.approx(0.1)

    # Test with None community prediction
    report = create_test_binary_report(
        prediction=0.6, community_prediction=None
    )
    assert report.deviation_points is None


def test_calculate_average_expected_log_score() -> None:
    reports = [
        create_test_binary_report(prediction=0.6, community_prediction=0.7),
        create_test_binary_report(prediction=0.3, community_prediction=0.4),
        create_test_binary_report(prediction=0.8, community_prediction=0.7),
    ]

    average_score = BinaryReport.calculate_average_expected_log_score(reports)
    assert isinstance(average_score, float)

    # Test with None community prediction
    reports_with_none = reports + [
        create_test_binary_report(prediction=0.5, community_prediction=None)
    ]
    with pytest.raises(AssertionError):
        BinaryReport.calculate_average_expected_log_score(reports_with_none)


def test_calculate_average_deviation_points() -> None:
    reports = [
        create_test_binary_report(
            prediction=0.6, community_prediction=0.7
        ),  # 0.1 deviation
        create_test_binary_report(
            prediction=0.3, community_prediction=0.4
        ),  # 0.1 deviation
        create_test_binary_report(
            prediction=0.8, community_prediction=0.7
        ),  # 0.1 deviation
    ]

    average_deviation = BinaryReport.calculate_average_deviation_points(
        reports
    )
    assert average_deviation == pytest.approx(0.1)

    # Test with None community prediction
    reports_with_none = reports + [
        create_test_binary_report(prediction=0.5, community_prediction=None)
    ]
    with pytest.raises(AssertionError):
        BinaryReport.calculate_average_deviation_points(reports_with_none)
