import logging
from typing import Any

from forecasting_tools.ai_models.ai_utils.ai_misc import (
    clean_indents,
    strip_code_block_markdown,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.helpers.configured_llms import BasicLlm
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_question import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


class FinalDecisionAgent:

    def __init__(
        self,
        research_as_markdown: str,
        question: MetaculusQuestion,
        number_of_predictions_to_run: int,
        cost_manager: MonetaryCostManager,
    ) -> None:
        assert (
            number_of_predictions_to_run > 0
        ), "Must run at least one prediction"
        assert research_as_markdown, "Research must be provided"
        self.research_as_markdown = research_as_markdown
        self.question = question
        self.report_type = ReportOrganizer.get_report_type_for_question_type(
            type(self.question)
        )
        self.number_of_predictions_to_run = number_of_predictions_to_run
        self.cost_manager = cost_manager
        self.__research_summary: str | None = None

    async def run_decision_agent(self) -> ForecastReport:
        try:
            research_summary = (
                await self.__get_research_summary_and_populate_if_empty()
            )
        except Exception as e:
            logger.error(f"Error in making research summary: {e}")
            research_summary = "Error in making research summary"

        logger.info(f"Running {self.number_of_predictions_to_run} predictions")
        final_prediction_coroutines = [
            self.report_type.run_prediction(self.question, research_summary)
            for _ in range(self.number_of_predictions_to_run)
        ]
        reasoned_predictions, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                final_prediction_coroutines
            )
        )
        if len(reasoned_predictions) == 0:
            raise ValueError("All forecasts errored")
        logger.info(
            f"{len(reasoned_predictions)} predictions successfully ran"
        )
        aggregated_prediction = await self.report_type.aggregate_predictions(
            [
                prediction.prediction_value
                for prediction in reasoned_predictions
            ]
        )
        explanation = await self.__create_unified_explanation(
            reasoned_predictions, aggregated_prediction
        )
        report = self.report_type(
            question=self.question,
            explanation=explanation,
            prediction=aggregated_prediction,
            price_estimate=self.cost_manager.current_usage,
        )
        logger.info("Compiled final report")
        return report

    async def __get_research_summary_and_populate_if_empty(self) -> str:
        if self.__research_summary:
            return self.__research_summary

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster working to summarize research you've done.
            Your goal is to summarize a research report the superforecasters have done.
            They have only 1min to read your summary, so make it concise and specific. Bring up signal not noise.

            # Instructions
            Please make a markdown report with three sections:
            1. Research Overview: Give 2 paragraphs summarazing the the research done. Surface things people would want to know for a forecast.
            2. Possible Base Rates: Make one bullet point for each unique possible base rate. Prioritize numbers, and do some calculations to find historical rates if possible (e.g. if you find there are 3 successful X out of 10 X total, then state your calculation and say 3 successful X out of 10 X total is 30% success rate).
            3. Pros Section: Make one bullet point for each unique pro. These should be inside view adjustments that would move your forecast up.
            4. Cons Section: Make one bullet point for each unique con. These should be outside view adjustments that would move your forecast down.

            Please cite from which question you got your information from (e.g. [Q2] for question 2, or [B1] for base rate question 1).
            Please try to prioritize things that people might miss on their own
            Don't use any of your own information. Only use information from the research report.

            # Compensation
            The superforecaster you work with is compensated based on the accuracy of their predictions, and you will get a cut of their compensation. You can make up to $1000 if you can increase their score with accurate information. Remember they are successful as much as they are properly able to assess uncertainty (i.e. overconfidence can be just as bad as underconfidence). Potray things as they are, and avoid misinformation, biased wording, point out potential biases in the data, etc.

            # Example
            An example is given in triple backticks below for the question of "Will North Korea conduct a nuclear test before 2025?".

            ```
            ### Research Overview
            Recent activities and statements by North Korean officials indicate a continued focus on their nuclear program, with plans to increase weapon-grade nuclear materials and arsenal. The nearly complete Experimental Light Water Reactor (ELWR) at Yongbyon could significantly boost plutonium production. North Korea has declared itself an "irreversible" nuclear power, preparing for a "real war" and threatening to turn the Pacific into a "firing range" [Q1]. These developments suggest a heightened commitment to advancing nuclear capabilities, potentially influencing the likelihood of a seventh nuclear test before 2025.
            North Korea has historically timed nuclear tests to coincide with significant international events, such as U.S. elections, to maximize attention and influence diplomatic dynamics. This pattern suggests a potential nuclear test around the 2024 U.S. Presidential Election to leverage international focus and influence the incoming administration's policy stance [Q2]. However, the Economist Intelligence Unit predicts North Korea may refrain from nuclear tests in 2023-24 due to economic dependence on China, which opposes further nuclear development [Q3].

            ### Possible Base Rates
            - Since North Korea began nuclear testing in 2006, it has conducted nuclear tests within three months of a U.S. Presidential Election on two occasions. There have been 4 U.S. Presidential Elections between 2006 and today (Oct 20 2024). 2/4 = 50% chance of a test within three months of a U.S. Presidential Election [B1].
            - Historically, North Korea has conducted six nuclear tests, with varying degrees of advance warning from credible sources [B2].
            - North Korea has a history of conducting nuclear tests following public statements about advancing their nuclear capabilities [B3].

            ### Pros
            - North Korea has a history of timing its nuclear tests and missile launches to coincide with significant international events, including U.S. elections, to maximize global attention and potentially influence diplomatic dynamics [Q2].
            - The president of South Korea said that he expects North Korea to conduct a nuclear test before 2025 [Q3].

            ### Cons
            - The Economist Intelligence Unit (EIU) predicts that North Korea may refrain from conducting nuclear tests in 2023-24 due to its economic dependence on China, which opposes further nuclear development [Q3].
            - Though there is a pattern of signs or reports of activity at nuclear sites, the exact timing of tests has been upredictable in the past and can vary between 0 to 3 months of the related indicating event if it happens at all [B2].
            ```

            # Question details
            Below are the details of the question the superforecaster is predicting on:

            {self.question.give_question_details_as_markdown()}


            # Research Report To Summarize
            Below in the triple back tick code block, is the research report you need to summarize.

            ```
            {self.research_as_markdown}
            ```

            Now please summarize the research report above using the markdown template given to you. Just fill in the template and give the markdown report, do not include any other text. Your summary will be published as is.
            """
        )
        model = BasicLlm(temperature=0)
        summary_markdown = await model.invoke(prompt)
        cleaned_summary_markdown = strip_code_block_markdown(summary_markdown)
        self.__research_summary = cleaned_summary_markdown
        logger.info(
            f"Made research summary: {cleaned_summary_markdown[:100]}..."
        )
        return cleaned_summary_markdown

    async def __create_unified_explanation(
        self,
        reasoned_predictions: list[ReasonedPrediction],
        aggregated_prediction: Any,
    ) -> str:
        assert self.__research_summary

        forecaster_prediction_bullet_points = ""
        for i, forecast in enumerate(reasoned_predictions):
            readable_prediction = self.report_type.make_readable_prediction(
                forecast.prediction_value
            )
            forecaster_prediction_bullet_points += (
                f"- *Forecaster {i + 1}*: {readable_prediction}\n"
            )

        combined_reasoning = ""
        for i, forecast in enumerate(reasoned_predictions):
            combined_reasoning += f"## Reasoning from forecaster {i + 1}\n"
            combined_reasoning += forecast.reasoning
            combined_reasoning += "\n\n"

        full_explanation_without_summary = clean_indents(
            f"""
            # SUMMARY
            *Question*: {self.question.question_text}\n
            *Final Prediction*: {self.report_type.make_readable_prediction(aggregated_prediction)}\n
            *Total Cost*: ${round(self.cost_manager.current_usage, 2)}

            ## Forecaster Team Summary
            {forecaster_prediction_bullet_points}

            {self.__research_summary}

            # RESEARCH
            {self.research_as_markdown}

            # FORECASTS
            {combined_reasoning}
            """
        )
        return full_explanation_without_summary
