from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class QuestionState(Enum):
    OPEN = "open"
    OTHER = "other"


class MetaculusQuestion(BaseModel, Jsonable):
    date_accessed: datetime = Field(default_factory=datetime.now)
    question_text: str
    question_id: int
    state: QuestionState
    resolution_criteria: str | None = None
    fine_print: str | None = None
    background_info: str | None = None
    page_url: str | None = None
    num_forecasters: int | None = None
    num_predictions: int | None = None
    close_time: datetime | None = None
    actual_resolution_time: datetime | None = None
    scheduled_resolution_time: datetime | None = None
    api_json: dict = Field(
        description="The API JSON response used to create the question",
        default_factory=dict,
    )

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> MetaculusQuestion:
        question_id = api_json["id"]
        json_state = api_json["status"]
        question_state = (
            QuestionState.OPEN if json_state == "open" else QuestionState.OTHER
        )
        scheduled_resolution_time = cls._parse_api_date(
            api_json["scheduled_resolve_time"]
        )
        resolution_time_is_in_past = scheduled_resolution_time < datetime.now()
        question_json: dict = api_json["question"]
        return MetaculusQuestion(
            state=question_state,
            question_text=question_json["title"],
            question_id=question_id,
            background_info=question_json.get("description", None),
            fine_print=question_json.get("fine_print", None),
            resolution_criteria=question_json.get("resolution_criteria", None),
            page_url=f"https://www.metaculus.com/questions/{question_id}",
            num_forecasters=api_json["nr_forecasters"],
            num_predictions=api_json["forecasts_count"],
            close_time=cls._parse_api_date(api_json["scheduled_close_time"]),
            actual_resolution_time=(
                scheduled_resolution_time
                if resolution_time_is_in_past
                else None
            ),  # TODO: Is the scheduled resolution time actually ever the 'actual' resolution time?
            scheduled_resolution_time=(
                scheduled_resolution_time
                if not resolution_time_is_in_past
                else None
            ),
            api_json=api_json,
        )

    @classmethod
    def _parse_api_date(cls, date_string: str) -> datetime:
        try:
            return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            pass
        try:
            return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
        return datetime.strptime(date_string, "%Y-%m-%d")

    @classmethod
    def get_api_type_name(cls) -> str:
        raise NotImplementedError(
            f"This function doesn't apply for base class {type(cls)}"
        )

    def give_question_details_as_markdown(self) -> str:
        today_string = datetime.now().strftime("%Y-%m-%d")
        question_details = textwrap.dedent(
            f"""
            You are trying to predict the outcome of the following question:
            {self.question_text}

            Here is the resolution criteria:
            {self.resolution_criteria}

            Here is the fine print:
            {self.fine_print}

            Here is some background information:
            {self.background_info}

            Today is (YYYY-MM-DD):
            {today_string}
            """
        )
        return question_details.strip()


class BinaryQuestion(MetaculusQuestion):
    community_prediction_at_access_time: float | None = None

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> BinaryQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        try:
            q2_center_community_prediction = api_json["question"]["aggregations"]["recency_weighted"]["latest"]["centers"]  # type: ignore
            assert len(q2_center_community_prediction) == 1
            community_prediction_at_access_time = (
                q2_center_community_prediction[0]
            )
        except (KeyError, TypeError):
            community_prediction_at_access_time = None
        return BinaryQuestion(
            community_prediction_at_access_time=community_prediction_at_access_time,
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> str:
        return "binary"


class DateQuestion(MetaculusQuestion):
    upper_bound: datetime
    lower_bound: datetime
    upper_bound_is_hard_limit: bool
    lower_bound_is_hard_limit: bool

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> DateQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        unparsed_upper_bound_date = api_json["question"]["possibilities"]["scale"]["max"]  # type: ignore
        unparsed_lower_bound_date = api_json["question"]["possibilities"]["scale"]["min"]  # type: ignore
        upper_bound_date = cls._parse_api_date(unparsed_upper_bound_date)
        lower_bound_date = cls._parse_api_date(unparsed_lower_bound_date)
        return DateQuestion(
            upper_bound=upper_bound_date,
            lower_bound=lower_bound_date,
            upper_bound_is_hard_limit=not api_json["question"]["open_upper_bound"],  # type: ignore
            lower_bound_is_hard_limit=not api_json["question"]["open_lower_bound"],  # type: ignore
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> str:
        return "date"


class NumericQuestion(MetaculusQuestion):
    upper_bound: float
    lower_bound: float
    upper_bound_is_hard_limit: bool
    lower_bound_is_hard_limit: bool

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> NumericQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        return NumericQuestion(
            upper_bound=api_json["question"]["possibilities"]["scale"]["max"],  # type: ignore
            lower_bound=api_json["question"]["possibilities"]["scale"]["min"],  # type: ignore
            upper_bound_is_hard_limit=not api_json["question"]["open_upper_bound"],  # type: ignore
            lower_bound_is_hard_limit=not api_json["question"]["open_lower_bound"],  # type: ignore
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> str:
        return "numeric"


class MultipleChoiceQuestion(MetaculusQuestion):
    options: list[str]

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> MultipleChoiceQuestion:
        normal_metaculus_question = super().from_metaculus_api_json(api_json)
        return MultipleChoiceQuestion(
            options=api_json["question"]["options"],  # type: ignore
            **normal_metaculus_question.model_dump(),
        )

    @classmethod
    def get_api_type_name(cls) -> str:
        return "multiple_choice"
