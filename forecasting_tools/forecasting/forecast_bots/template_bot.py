import re
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.perplexity import Perplexity
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.configured_llms import AdvancedLlm
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOption,
    PredictedOptionSet,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
    Percentile,
)


class TemplateBot(ForecastBot):
    FINAL_DECISION_LLM = AdvancedLlm(temperature=0.7)

    async def run_research(self, question: MetaculusQuestion) -> str:
        system_prompt = clean_indents(
            """
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.
            """
        )
        prompt = clean_indents(
            f"""
            Please give a concise research report on the below question given the below context:
            Question:
            {question.question_text}

            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        response = await Perplexity(system_prompt=system_prompt).invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
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
        gpt_forecast = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(gpt_forecast)
        reasoning = (
            gpt_forecast
            + "\nThe original forecast may have been clamped between 5% and 95%."
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionSet]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        gpt_forecast = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_multiple_choice_rationale(
            gpt_forecast, question.options
        )
        reasoning = gpt_forecast
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
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
            Provide at values for your 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% estimates.
            The lowest value should have a probability approaching 10%, and the highest value should approach 90%.
            Remember that its very easy to be overconfident. 10% should feel like "this couldn't possibly get below this number!", and probability of 90% should feel like "There is not chance this will get anywhere above this number!"
            """
        )
        reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_numeric_rationale(
            reasoning, question
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _extract_forecast_from_binary_rationale(self, rationale: str) -> float:
        max_prediction = 95
        min_prediction = 5
        matches = re.findall(r"(\d+)%", rationale)
        if matches:
            # Return the last number found before a '%'
            original_number = int(matches[-1])
            clamped_number = min(
                max_prediction, max(min_prediction, original_number)
            )
            assert (
                min_prediction <= clamped_number <= max_prediction
            ), f"Clamped number {clamped_number} is not between {min_prediction} and {max_prediction}"
            return clamped_number / 100
        else:
            raise ValueError(
                f"Could not extract prediction from response: {rationale}"
            )

    def _extract_forecast_from_multiple_choice_rationale(
        self, reasoning: str, options: list[str]
    ) -> PredictedOptionSet:
        option_probabilities = []
        # Iterate through each line in the text
        for expected_option in options:
            probability_found = False
            for line in reasoning.split("\n"):
                if expected_option in line:
                    # Extract all numbers from the line
                    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", line)
                    numbers_no_commas = [
                        num.replace(",", "") for num in numbers
                    ]
                    # Convert strings to float or int
                    numbers = [
                        float(num) if "." in num else int(num)
                        for num in numbers_no_commas
                    ]
                    if len(numbers) >= 1:
                        last_number = numbers[-1]
                        option_probabilities.append(last_number)
                        probability_found = True
                        break

            assert (
                probability_found
            ), f"No probability found for option: {expected_option}"

        assert len(option_probabilities) == len(
            options
        ), f"Number of option probabilities {len(option_probabilities)} does not match number of options {len(options)}"

        total_sum = sum(option_probabilities)
        decimal_list = [x / total_sum for x in option_probabilities]

        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in decimal_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        normalized_option_probabilities = normalized_list

        predicted_options: list[PredictedOption] = []
        for i in range(len(options)):
            predicted_options.append(
                PredictedOption(
                    option_name=options[i],
                    probability=normalized_option_probabilities[i],
                )
            )

        return PredictedOptionSet(predicted_options=predicted_options)

    def _extract_forecast_from_numeric_rationale(
        self, reasoning: str, question: NumericQuestion
    ) -> NumericDistribution:
        matches = re.findall(
            r"Probability of a value below (\d+(?:\.\d+)?) is (\d+(?:\.\d+)?)%",
            reasoning,
        )
        if matches:
            percentiles = []
            for value, percentile in matches:
                percentiles.append(
                    Percentile(
                        value=float(value.replace(",", "")),
                        percentile=float(percentile) / 100,
                    )
                )
            return NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=question.zero_point,
            )
        else:
            raise ValueError(
                f"Could not extract prediction from response: {reasoning}"
            )
