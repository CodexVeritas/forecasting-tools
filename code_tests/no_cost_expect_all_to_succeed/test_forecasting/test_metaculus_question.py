from code_tests.utilities_for_tests import jsonable_assertations
from forecasting_tools.forecasting.questions_and_reports.metaculus_question import (
    BinaryQuestion,
)


def test_metaculus_question_is_jsonable() -> None:
    temp_writing_path = "temp/temp_metaculus_question.json"
    read_report_path = "code_tests/no_cost_expect_all_to_succeed/test_forecasting/forecasting_test_data/metaculus_questions.json"
    jsonable_assertations.assert_reading_and_printing_from_file_works(
        BinaryQuestion, read_report_path, temp_writing_path
    )
