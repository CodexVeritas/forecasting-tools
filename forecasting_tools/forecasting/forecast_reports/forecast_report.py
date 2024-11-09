from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, field_validator

from forecasting_tools.forecasting.forecast_reports.metaculus_question import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.forecast_reports.report_section import (
    ReportSection,
)
from forecasting_tools.util.jsonable import Jsonable

T = TypeVar("T")


class ReasonedPrediction(BaseModel, Generic[T]):
    prediction_value: T
    reasoning: str


class ForecastReport(BaseModel, Jsonable, ABC):
    question: MetaculusQuestion
    explanation: str
    other_notes: str | None = None
    price_estimate: float | None = None
    forecast_info: list[Any] = []
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
    async def aggregate_predictions(cls, predictions: list[T]) -> T:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @classmethod
    @abstractmethod
    async def run_prediction(
        cls, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[Any]:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @classmethod
    @abstractmethod
    def make_readable_prediction(cls, prediction: Any) -> str:
        raise NotImplementedError(
            "Subclass must implement this abstract method"
        )

    @classmethod
    async def combine_report_list_into_one(
        cls,
        reports: list[ForecastReport],
        final_cost: float,
        duration_in_minutes: float,
    ) -> ForecastReport:
        assert len(reports) > 0, "No reports were provided"
        first_report = reports[0]
        assert all(
            isinstance(report, type(first_report)) for report in reports
        ), "All reports must be of the same type"
        assert all(
            report.question == first_report.question for report in reports
        ), "All reports must be for the same question"
        if len(reports) == 1:
            return reports[0]

        report_type = type(reports[0])
        aggregated_prediction = await report_type.aggregate_predictions(
            [report.prediction for report in reports]
        )
        combined_explanation = await cls.__create_combined_explanation(
            reports, final_cost, duration_in_minutes
        )
        combined_report = report_type(
            question=reports[0].question,
            explanation=combined_explanation,
            prediction=aggregated_prediction,
            forecast_info=[report for report in reports],
            price_estimate=final_cost,
        )
        return combined_report

    @classmethod
    async def __create_combined_explanation(
        cls,
        reports: list[ForecastReport],
        final_cost: float,
        duration_in_minutes: float,
    ) -> str:
        combined_sections: dict[str, list[str]] = {
            "Summary": [],
            "Research": [],
            "Forecasts and Rationales": [],
        }

        for report in reports:
            combined_sections["Summary"].append(report.summary)
            combined_sections["Research"].append(report.research)
            combined_sections["Forecasts and Rationales"].append(
                report.forecast_rationales
            )

        combined_explanation = ""
        for section_title, section_contents in combined_sections.items():
            combined_explanation += f"# {section_title}\n\n"

            if section_title == "Summary":
                combined_explanation += await cls.__combine_summary_sections(
                    reports, section_contents, final_cost, duration_in_minutes
                )
            else:
                combined_explanation += (
                    await cls.__combine_non_summary_sections(section_contents)
                )
        return combined_explanation.strip()

    @classmethod
    async def __combine_summary_sections(
        cls,
        reports: list[ForecastReport],
        section_contents: list[str],
        final_cost: float,
        duration_in_minutes: float,
    ) -> str:
        combined_explanation = ""
        report_type = type(reports[0])
        aggregated_prediction = await report_type.aggregate_predictions(
            [report.prediction for report in reports]
        )
        final_prediction = report_type.make_readable_prediction(
            aggregated_prediction
        )
        combined_explanation += f"Final Cost: ${final_cost:.2f}\n\n"
        combined_explanation += f"Final Prediction: {final_prediction}\n\n"
        combined_explanation += (
            f"Time to run: {duration_in_minutes:.2f} minutes\n\n"
        )
        for report_num, content in enumerate(section_contents, 1):
            # Remove the top-level heading from each summary
            lines = content.split("\n")
            content_without_heading = "\n".join(lines[1:])

            # Increase heading level for summary sections
            modified_content = f"## Report {report_num}: Summary\n"
            for line in content_without_heading.split("\n"):
                if line.startswith("#"):
                    line = (
                        "#" + line
                    )  # Add one more # to increase heading level
                modified_content += line + "\n"

            combined_explanation += modified_content + "\n\n"
        return combined_explanation

    @classmethod
    async def __combine_non_summary_sections(
        cls,
        section_contents: list[str],
    ) -> str:
        combined_explanation = ""
        for report_num, content in enumerate(section_contents, 1):
            # Remove the top-level heading from each section
            lines = content.split("\n")
            content_without_heading = "\n".join(lines[1:])

            # Add report number to each h2 heading
            modified_content = ""
            for line in content_without_heading.split("\n"):
                if line.startswith("## "):
                    line = f"## R{report_num}: {line[3:]}"
                modified_content += line + "\n"

            combined_explanation += modified_content + "\n\n"
        return combined_explanation

    def _get_section_content(self, index: int, expected_word: str) -> str:
        if len(self.report_sections) <= index:
            raise ValueError(f"Report must have at least {index + 1} sections")
        content = self.report_sections[index].text_of_section_and_subsections
        first_line = content.split("\n")[0]
        if expected_word not in first_line.lower():
            raise ValueError(
                f"Section must contain the word '{expected_word}'"
            )
        return content
