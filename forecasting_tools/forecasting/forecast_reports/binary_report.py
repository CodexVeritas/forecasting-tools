from __future__ import annotations

import re
import statistics
from datetime import datetime

from pydantic import AliasChoices, Field, field_validator

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.forecasting.forecast_reports.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
)
from forecasting_tools.forecasting.forecast_reports.metaculus_question import (
    BinaryQuestion,
)
from forecasting_tools.forecasting.helpers.configured_llms import AdvancedLlm
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi


class BinaryReport(ForecastReport):
    question: BinaryQuestion
    prediction: float = Field(
        validation_alias=AliasChoices("prediction_in_decimal", "prediction")
    )

    @field_validator("prediction")
    def validate_prediction(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Prediction must be between 0 and 1")
        return v

    async def publish_report_to_metaculus(self) -> None:
        MetaculusApi.post_binary_question_prediction(
            self.question.question_id, self.prediction
        )
        MetaculusApi.post_question_comment(
            self.question.question_id, self.explanation
        )

    @classmethod
    async def aggregate_predictions(cls, predictions: list[float]) -> float:
        for prediction in predictions:
            assert 0 <= prediction <= 1, "Predictions must be between 0 and 1"
            assert isinstance(prediction, float), "Predictions must be floats"
        return statistics.median(predictions)

    @classmethod
    def make_readable_prediction(cls, prediction: float) -> str:
        return f"{round(prediction * 100, 2)}%"

    @property
    def community_prediction(self) -> float | None:
        return self.question.community_prediction_at_access_time

    @property
    def deviation_score(self) -> float | None:
        community_prediction = self.community_prediction
        if community_prediction is None:
            return None
        return abs(community_prediction - self.prediction) ** 2

    @staticmethod
    def calculate_average_deviation_score(
        reports: list[BinaryReport],
    ) -> float:
        deviation_scores: list[float | None] = [
            report.deviation_score for report in reports
        ]
        validated_deviation_scores: list[float] = []
        for score in deviation_scores:
            assert score is not None
            validated_deviation_scores.append(score)
        average_deviation_score = sum(validated_deviation_scores) / len(
            validated_deviation_scores
        )
        return average_deviation_score

    @classmethod
    async def run_prediction(
        cls,
        question: BinaryQuestion,
        research: str,
    ) -> ReasonedPrediction[float]:
        assert isinstance(
            question, BinaryQuestion
        ), "Question must be a BinaryQuestion"
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}

            Background information:
            {question.background_info if question.background_info else "No background information provided."}

            Resolution criteria:
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

            Fine print:
            {question.fine_print if question.fine_print else "No fine print provided."}


            Your research assistant says:
            ```
            {research}
            ```

            Today is {datetime.now().strftime("%Y-%m-%d")}.


            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) What the outcome would be if nothing changed.
            (c) The most important factors that will influence a successful/unsuccessful resolution.
            (d) What do you not know that should give you pause and lower confidence? Remember people are statistically overconfident.
            (e) What you would forecast if you were to only use historical precedent (i.e. how often this happens in the past) without any current information.
            (f) What you would forecast if there was only a quarter of the time left.
            (g) What you would forecast if there was 4x the time left.

            You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        final_prediction_model = AdvancedLlm()
        gpt_forecast = await final_prediction_model.invoke(prompt)
        prediction, clamped_message = cls.__extract_prediction_from_response(
            gpt_forecast, max_prediction=95, min_prediction=5
        )
        reasoning = gpt_forecast + clamped_message
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    @classmethod
    def __extract_prediction_from_response(
        cls, forecast_text: str, max_prediction: int, min_prediction: int
    ) -> tuple[float, str]:
        assert (
            1 <= max_prediction <= 99
        ), "Max prediction must be between 1 and 99"
        assert (
            1 <= min_prediction <= 99
        ), "Min prediction must be between 1 and 99"
        matches = re.findall(r"(\d+)%", forecast_text)
        if matches:
            # Return the last number found before a '%'
            original_number = int(matches[-1])
            clamped_number = min(
                max_prediction, max(min_prediction, original_number)
            )
            clamped_message = ""
            if clamped_number != original_number:
                clamped_message = f"\n\nNote: The original forecast of {original_number}% was clamped to {clamped_number}%."
            assert min_prediction <= clamped_number <= max_prediction
            return clamped_number / 100, clamped_message
        else:
            raise ValueError(
                f"Could not extract prediction from response: {forecast_text}"
            )
