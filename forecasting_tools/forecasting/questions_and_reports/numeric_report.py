import logging
import random
import re
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.forecasting.helpers.configured_llms import BasicLlm
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    NumericQuestion,
)

logger = logging.getLogger(__name__)


class Percentile(BaseModel):
    value: float
    percentile: float


class NumericDistribution(BaseModel):
    percentiles: list[Percentile]

    @property
    def cdf(self) -> list[float]:
        raise NotImplementedError("Not implemented")


class NumericReport(ForecastReport):
    question: NumericQuestion
    prediction: NumericDistribution

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[NumericDistribution]
    ) -> NumericDistribution:
        logger.warning(
            "This aggregation strategy sucks and should be temporary till format is finalized"
        )
        # Take a random prediction from the list

        if not predictions:
            raise ValueError("No predictions to aggregate")

        return random.choice(predictions)

    @classmethod
    async def run_prediction(
        cls, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            ```
            {research}
            ```

            The question implies that the majority of the probability distribution is probably in the range:
            {question.lower_bound} to {question.upper_bound}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) What the outcome would be if nothing changed.
            (c) What you would forecast if there was only a quarter of the time left.
            (d) What you would forecast if there was 4x the time left.

            You write your rationale and then the last thing you write is your final answer as a series of probability distributions.
            Each line should be in the format: "Probability of a value below Y is X%". Make sure to use this EXACT format, and change out only Y and X.
            Provide at least 3 and up to 10 such lines, ensuring the probabilities increase as the values increase.
            The lowest value should have a probability approaching 10%, and the highest value should approach 90%.
            Remember that its very easy to be overconfident. 10% should feel like "this couldn't possibly get below this number!", and probability of 90% should feel like "There is not chance this will get anywhere above this number!"
            """
        )
        final_prediction_model = BasicLlm(temperature=0.7)
        gpt_forecast = await final_prediction_model.invoke(prompt)
        prediction = cls.__extract_prediction_from_response(gpt_forecast)
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=gpt_forecast
        )

    @classmethod
    def __extract_prediction_from_response(
        cls, forecast_text: str
    ) -> NumericDistribution:
        matches = re.findall(
            r"Probability of a value below (\d+(?:\.\d+)?) is (\d+(?:\.\d+)?)%",
            forecast_text,
        )
        if matches:
            prediction = []
            prev_prob = 0.0
            for value, cum_prob in matches:
                cum_prob = float(cum_prob) / 100
                prob = cum_prob - prev_prob
                prediction.append((float(value), prob))
                prev_prob = cum_prob
            return prediction
        else:
            raise ValueError(
                f"Could not extract prediction from response: {forecast_text}"
            )

    @classmethod
    def make_readable_prediction(cls, prediction: NumericDistribution) -> str:
        sorted_prediction = sorted(prediction, key=lambda x: x[0])
        cumulative_prob = 0
        readable = "Probability distribution:\n"
        for value, prob in sorted_prediction:
            cumulative_prob += prob
            readable += f"  Probability of value below {value}: {cumulative_prob:.2%}\n"
        return readable

    async def publish_report_to_metaculus(self) -> None:
        raise NotImplementedError("Format still TBD")
