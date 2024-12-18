from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, field_validator

from forecasting_tools.forecasting.questions_and_reports.questions import (
    Question,
)
from forecasting_tools.forecasting.questions_and_reports.report_section import (
    ReportSection,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ReasonedPrediction(BaseModel, Generic[T]):
    prediction_value: T
    reasoning: str


class ResearchWithPredictions(BaseModel, Generic[T]):
    research_report: str
    summary_report: str
    predictions: list[ReasonedPrediction[T]]


class ForecastReport(BaseModel, Jsonable, ABC):
    question: Question
    explanation: str
    other_notes: str | None = None
    price_estimate: float | None = None
    minutes_taken: float | None = None
    prediction: Any

    @field_validator("explanation")
    @classmethod
    def validate_explanation_starts_with_hash(cls, v: str) -> str:
        if not v.strip().startswith("#"):
            raise ValueError("Explanation must start with a '#' character")
        return v

    @property
    def report_sections(self) -> list[ReportSection]:
        return ReportSection.turn_markdown_into_report_sections(
            self.explanation
        )

    @property
    def summary(self) -> str:
        return self._get_section_content(index=0, expected_word="summary")

    @property
    def research(self) -> str:
        return self._get_section_content(index=1, expected_word="research")

    @property
    def forecast_rationales(self) -> str:
        return self._get_section_content(index=2, expected_word="forecast")

    @abstractmethod
    async def publish_report_to_metaculus(self) -> None:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @classmethod
    @abstractmethod
    async def aggregate_predictions(
        cls, predictions: list[T], question: Question
    ) -> T:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @classmethod
    @abstractmethod
    def make_readable_prediction(cls, prediction: Any) -> str:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    def _get_section_content(self, index: int, expected_word: str) -> str:
        if len(self.report_sections) <= index:
            raise ValueError(f"Report must have at least {index + 1} sections")
        content = self.report_sections[index].text_of_section_and_subsections
        first_line = content.split("\n")[0]
        if expected_word.lower() not in first_line.lower():
            raise ValueError(
                f"Section must contain the word '{expected_word}'"
            )
        return content
