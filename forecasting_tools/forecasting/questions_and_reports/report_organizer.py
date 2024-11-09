from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.date_report import (
    DateReport,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_question import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericReport,
)


class ReportOrganizer:
    __TYPE_MAPPING = [
        {
            "type": BinaryQuestion,
            "test_question_id": 384,  # https://www.metaculus.com/questions/384/
            "report_type": BinaryReport,
        },
        {
            "type": NumericQuestion,
            "test_question_id": 26253,  # https://www.metaculus.com/questions/26253/
            "report_type": NumericReport,
        },
        {
            "type": DateQuestion,
            "test_question_id": 5121,  # https://www.metaculus.com/questions/5121/
            "report_type": DateReport,
        },
        {
            "type": MultipleChoiceQuestion,
            "test_question_id": 21465,  # https://www.metaculus.com/questions/21465/
            "report_type": MultipleChoiceReport,
        },
    ]

    @classmethod
    def get_example_question_id_for_question_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> int:
        assert issubclass(question_type, MetaculusQuestion)
        for question_info in cls.__TYPE_MAPPING:
            if question_info["type"] == question_type:
                return question_info["test_question_id"]
        raise ValueError(f"No question ID found for type {question_type}")

    @classmethod
    def get_report_type_for_question_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> type[ForecastReport]:
        assert issubclass(question_type, MetaculusQuestion)
        for question_info in cls.__TYPE_MAPPING:
            if question_info["type"] == question_type:
                return question_info["report_type"]
        raise ValueError(f"No report type found for type {question_type}")

    @classmethod
    def get_live_example_question_of_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> MetaculusQuestion:
        assert issubclass(question_type, MetaculusQuestion)
        question_id = cls.get_example_question_id_for_question_type(
            question_type
        )
        question = MetaculusApi.get_question_by_id(question_id)
        assert isinstance(question, question_type)
        return question

    @classmethod
    def get_all_report_types(cls) -> list[type[ForecastReport]]:
        return [
            question_info["report_type"]
            for question_info in cls.__TYPE_MAPPING
        ]

    @classmethod
    def get_all_question_types(cls) -> list[type[MetaculusQuestion]]:
        return [question_info["type"] for question_info in cls.__TYPE_MAPPING]
