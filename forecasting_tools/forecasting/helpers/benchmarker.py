import subprocess
from datetime import datetime
from typing import Literal

import typeguard

from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)


class Benchmarker:

    @classmethod
    async def benchmark_forecast_bot(
        cls,
        forecast_bot: ForecastBot,
        number_of_questions_to_test: (
            Literal["shallow", "medium", "deep"] | int
        ),
        file_path_to_save_reports: str = "logs/forecasts/benchmarks/",
    ) -> float:
        """
        Below are the conclusions of a rough (and potentially flawed) simulation of tournaments and skill levels
        to help with choosing sample sizes. See https://chatgpt.com/share/3fbc8106-829d-4fb3-a9e6-af0badf266df

        lower = decent but lower quality = 50% of my deviation values (prediction - community vote) is above ~0.25
        higher = decent but higher quality = 50% of my devaiation values (predition - community vote) is above ~0.17

        At 10 samples
        - 20% of being lower, but seeming higher
        - 40% chance of being lower but seeming higher

        At 20 samples
        - 5% of being lower, but seeming higher
        - 20% being higher, but seeming lower

        At 30 samples
        - 3% of being lower, but seeming higher
        - 10% of being higher, but seeming lower

        The chances for misidentification decreases as the bot gains a deviation distribution that leans more towards 0. The chances get higher as it leans more towards 1.
        TODO: Incorporate the following analysis which is more rigorous. TLDR the numbers above are probably wrong: https://forum.effectivealtruism.org/posts/DzqSh7akX28JEHf9H/comparing-two-forecasters-in-an-ideal-world
        """

        if isinstance(number_of_questions_to_test, int):
            num_questions_to_benchmark_on = number_of_questions_to_test
        elif number_of_questions_to_test == "shallow":
            num_questions_to_benchmark_on = 10
        elif number_of_questions_to_test == "medium":
            num_questions_to_benchmark_on = 30
        elif number_of_questions_to_test == "deep":
            num_questions_to_benchmark_on = 100

        questions = MetaculusApi.get_benchmark_questions(
            num_questions_to_benchmark_on,
            random_seed=42,
            # Choose a random seed so all benchmarks in a similar time period use the same questions
        )
        assert len(questions) == num_questions_to_benchmark_on
        questions = typeguard.check_type(questions, list[MetaculusQuestion])
        reports = await forecast_bot.forecast_questions(questions)
        reports = typeguard.check_type(reports, list[BinaryReport])
        average_deviation_score = (
            BinaryReport.calculate_average_expected_log_score(reports)
        )
        rounded_score = round(average_deviation_score, 4)

        git_hash = cls.__get_git_commit_hash()
        if not file_path_to_save_reports.endswith("/"):
            file_path_to_save_reports += "/"
        file_path_to_save_reports = (
            f"{file_path_to_save_reports}"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__"
            f"score_{rounded_score}__"
            f"git_{git_hash}__"
            f".json"
        )
        BinaryReport.save_object_list_to_file_path(
            reports, file_path_to_save_reports
        )
        return average_deviation_score

    @classmethod
    def __get_git_commit_hash(cls) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"]
                )
                .decode("ascii")
                .strip()
            )
        except Exception:
            return "no_git_hash"
