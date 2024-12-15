from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.questions_and_reports.metaculus_questions import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.sub_question_researchers.research_coordinator import (
    ResearchCoordinator,
)


class MainBot(TemplateBot):

    def __init__(
        self,
        research_reports_per_question: int = 3,
        predictions_per_research_report: int = 5,
        use_research_summary_to_forecast: bool = True,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        number_of_background_questions_to_ask: int = 5,
        number_of_base_rate_questions_to_ask: int = 5,
        number_of_base_rates_to_do_deep_research_on: int = 0,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
        )
        self.number_of_background_questions_to_ask = (
            number_of_background_questions_to_ask
        )
        self.number_of_base_rate_questions_to_ask = (
            number_of_base_rate_questions_to_ask
        )
        self.number_of_base_rates_to_do_deep_research_on = (
            number_of_base_rates_to_do_deep_research_on
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        research_manager = ResearchCoordinator(question)
        combined_markdown = (
            await research_manager.create_full_markdown_research_report(
                self.number_of_background_questions_to_ask,
                self.number_of_base_rate_questions_to_ask,
                self.number_of_base_rates_to_do_deep_research_on,
            )
        )
        return combined_markdown

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        research_coordinator = ResearchCoordinator(question)
        summary_report = (
            await research_coordinator.summarize_full_research_report(research)
        )
        return summary_report
