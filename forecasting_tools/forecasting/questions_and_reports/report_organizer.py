from pydantic import BaseModel

from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
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


class TypeMapping(BaseModel):
    question_type: type[MetaculusQuestion]
    test_question_id: int
    report_type: type[ForecastReport] | None


class ReportOrganizer:
    __TYPE_MAPPING = [
        TypeMapping(
            question_type=BinaryQuestion,
            test_question_id=384,  # https://www.metaculus.com/questions/384/
            report_type=BinaryReport,
        ),
        TypeMapping(
            question_type=NumericQuestion,
            test_question_id=26253,  # https://www.metaculus.com/questions/26253/
            report_type=NumericReport,
        ),
        TypeMapping(
            question_type=DateQuestion,
            test_question_id=5121,  # https://www.metaculus.com/questions/5121/
            report_type=None,
        ),
        TypeMapping(
            question_type=MultipleChoiceQuestion,
            test_question_id=21465,  # https://www.metaculus.com/questions/21465/
            report_type=MultipleChoiceReport,
        ),
    ]

    @classmethod
    def get_example_question_id_for_question_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> int:
        assert issubclass(question_type, MetaculusQuestion)
        for mapping in cls.__TYPE_MAPPING:
            if mapping.question_type == question_type:
                return mapping.test_question_id
        raise ValueError(f"No question ID found for type {question_type}")

    @classmethod
    def get_report_type_for_question_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> type[ForecastReport]:
        assert issubclass(question_type, MetaculusQuestion)
        for mapping in cls.__TYPE_MAPPING:
            if mapping.question_type == question_type:
                if mapping.report_type is None:
                    raise ValueError(
                        f"No report type found for type {question_type}"
                    )
                return mapping.report_type
        raise ValueError(f"No report type found for type {question_type}")

    @classmethod
    def get_live_example_question_of_type(
        cls, question_type: type[MetaculusQuestion]
    ) -> MetaculusQuestion:
        assert issubclass(question_type, MetaculusQuestion)
        question_id = cls.get_example_question_id_for_question_type(
            question_type
        )
        question = MetaculusApi.get_question_by_post_id(question_id)
        assert isinstance(question, question_type)
        return question

    @classmethod
    def get_all_report_types(cls) -> list[type[ForecastReport]]:
        return [
            mapping.report_type
            for mapping in cls.__TYPE_MAPPING
            if mapping.report_type is not None
        ]

    @classmethod
    def get_all_question_types(cls) -> list[type[MetaculusQuestion]]:
        return [mapping.question_type for mapping in cls.__TYPE_MAPPING]
