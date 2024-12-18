from pydantic import BaseModel

from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericReport,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    Question,
)


class TypeMapping(BaseModel):
    question_type: type[Question]
    test_post_id: int
    report_type: type[ForecastReport] | None


class ReportOrganizer:
    __TYPE_MAPPING = [
        TypeMapping(
            question_type=BinaryQuestion,
            test_post_id=578,  # https://www.metaculus.com/questions/578/human-extinction-by-2100/
            report_type=BinaryReport,
        ),
        TypeMapping(
            question_type=NumericQuestion,
            test_post_id=14333,  # https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
            report_type=NumericReport,
        ),
        # TypeMapping(
        #     question_type=DateQuestion,
        #     test_question_id=4110,  # https://www.metaculus.com/questions/4110/birthdate-of-oldest-living-human-in-2200/
        #     report_type=None,
        # ),
        TypeMapping(
            question_type=MultipleChoiceQuestion,
            test_post_id=22427,  # https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
            report_type=MultipleChoiceReport,
        ),
    ]

    @classmethod
    def get_example_question_id_for_question_type(
        cls, question_type: type[Question]
    ) -> int:
        assert issubclass(question_type, Question)
        for mapping in cls.__TYPE_MAPPING:
            if mapping.question_type == question_type:
                return mapping.test_post_id
        raise ValueError(f"No question ID found for type {question_type}")

    @classmethod
    def get_report_type_for_question_type(
        cls, question_type: type[Question]
    ) -> type[ForecastReport]:
        assert issubclass(question_type, Question)
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
        cls, question_type: type[Question]
    ) -> Question:
        assert issubclass(question_type, Question)
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
    def get_all_question_types(cls) -> list[type[Question]]:
        return [mapping.question_type for mapping in cls.__TYPE_MAPPING]
