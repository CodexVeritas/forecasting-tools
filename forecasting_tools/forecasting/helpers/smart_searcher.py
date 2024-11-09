import asyncio
import logging
import re
import urllib.parse
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.basic_model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.basic_model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.ai_models.exa_searcher import (
    ExaHighlightQuote,
    ExaSearcher,
    SearchInput,
)
from forecasting_tools.forecasting.helpers.configured_llms import BasicLlm
from forecasting_tools.forecasting.helpers.url_scraper import UrlScraper
from forecasting_tools.forecasting.helpers.works_cited_creator import (
    WorksCitedCreator,
)
from forecasting_tools.util.async_batching import (
    run_coroutines_while_removing_and_logging_exceptions,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)
Base64Image = str


class ScreenshotDescription(BaseModel, Jsonable):
    image_description: str
    url: str
    title: str | None
    readable_publish_date: str | None
    image_data: Base64Image


class SmartSearcher(OutputsText, AiModel):
    """
    Answers a prompt, using search results to inform its response.
    """

    def __init__(
        self,
        *args,
        temperature: float = 0,
        include_works_cited_list: bool = False,
        use_brackets_around_citations: bool = True,
        num_searches_to_run: int = 2,
        num_sites_to_deep_dive: int = 0,
        num_sites_per_search: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert 0 <= temperature <= 1, "Temperature must be between 0 and 1"
        assert (
            num_searches_to_run >= 1
        ), "Number of searches to run must be at least 1"
        assert (
            num_sites_per_search >= 1
        ), "Number of sites per search must be at least 1"
        assert (
            0
            <= num_sites_to_deep_dive
            <= num_sites_per_search * num_searches_to_run
        ), f"Number of sites to screenshot must be at least 0 and at most {num_sites_per_search * num_searches_to_run}"
        self.temperature = temperature
        self.num_quotes_to_evaluate_from_search = 15
        self.number_of_searches_to_run = num_searches_to_run
        self.exa_searcher = ExaSearcher(
            include_text=False,
            include_highlights=True,
            num_results=num_sites_per_search,
        )
        self.llm = BasicLlm(temperature=temperature)
        self.include_works_cited_list = include_works_cited_list
        self.use_citation_brackets = use_brackets_around_citations
        self.num_sites_to_screenshot = num_sites_to_deep_dive

    async def invoke(self, prompt: str) -> str:
        logger.debug(f"Running search for prompt: {prompt}")
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        report, _ = await self._mockable_direct_call_to_model(prompt)
        logger.debug(f"Report: {report[:1000]}...")
        return report

    async def _mockable_direct_call_to_model(
        self, prompt: str
    ) -> tuple[str, list[ExaHighlightQuote]]:
        searches = await self.__come_up_with_search_queries(prompt)
        quotes = await self.__search_for_quotes(searches)
        image_descriptions = await self.__get_image_descriptions(
            quotes, prompt
        )
        report = await self.__compile_report(
            quotes, prompt, image_descriptions
        )
        if self.include_works_cited_list:
            works_cited_list = WorksCitedCreator.create_works_cited_list(
                quotes, report
            )
            report = report + "\n\n" + works_cited_list
        report = self.__add_quote_links_to_citations(report, quotes)
        report = self.__add_image_links_to_citations(
            report, image_descriptions
        )
        return report, quotes

    async def __come_up_with_search_queries(
        self, prompt: str
    ) -> list[SearchInput]:
        prompt = clean_indents(
            f"""
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><>START INSTRUCTIONS<><><><><><>
            {prompt}
            <><><><><><>END INSTRUCTIONS<><><><><><>

            Generate {self.number_of_searches_to_run} google searches that will help you fulfill any questions in the instructions.
            Consider and walk through the following before giving your json answers:
            - What are some possible search queries and strategies that would be useful?
            - What are the aspects of the question that are most important? Are there multiple aspects?
            - Where is the information you need likely to be found or what will good sources likely contain in them?
                - Wikipedia is good for lists of things and events
                - FRED (Economic Research at the St. Louis Fed) is good for economic data
                - Our World in Data
                - OECD Stats
                - UN Data
                - Data.gov
                - Yahoo Finance
                - Bureau of Economic Analysis
                - Consider things specific to the area you are researching
            - Would it already be at the top of the search results, or should you filter for it?
            - What filters would help you achieve this to increase the information density of the results? Consider:
                - Please only use the additional search fields ONLY IF it would return useful results or you are specifically instructed to do so.
                - Unless you have a very specific site in mind, or a have a more than 2 queries, you should probably only use the date filter
                - Only use the date filter if it makes sense to do so. Don't unnecessarily exclude older content.
                - Consider if you can get the same results by putting the keywords in the search query instead (since in may ways this acts as a filter even if its not explicit)
                - Please don't put anything in the highlight query unless requested to do so.
            - You have limited searches, which approaches would be highest priority?
            Remember today is {datetime.now().strftime("%Y-%m-%d")}.

            {self.llm.get_schema_format_instructions_for_pydantic_type(SearchInput)}

            Make sure to return a list of the search inputs as a list of JSON objects in this schema.
            Do not give the json in separate chunks. It needs to be in one combined list.
            """
        )
        search_terms = await self.llm.invoke_and_return_verified_type(
            prompt, list[SearchInput]
        )
        search_log = "\n".join(
            [
                f"Search {i+1}: {search}"
                for i, search in enumerate(search_terms)
            ]
        )
        logger.info(f"Decided on searches:\n{search_log}")
        return search_terms

    async def __search_for_quotes(
        self, search_inputs: list[SearchInput]
    ) -> list[ExaHighlightQuote]:
        all_quotes: list[list[ExaHighlightQuote]] = await asyncio.gather(
            *[
                self.exa_searcher.invoke_for_highlights_in_relevance_order(
                    search
                )
                for search in search_inputs
            ]
        )
        flattened_quotes = [
            quote for sublist in all_quotes for quote in sublist
        ]
        unique_quotes: dict[str, ExaHighlightQuote] = {}
        for quote in flattened_quotes:
            if quote.highlight_text not in unique_quotes:
                unique_quotes[quote.highlight_text] = quote
        deduplicated_quotes = sorted(
            unique_quotes.values(), key=lambda x: x.score, reverse=True
        )
        if len(deduplicated_quotes) == 0:
            raise RuntimeError("No quotes found")
        if len(deduplicated_quotes) < self.num_quotes_to_evaluate_from_search:
            logger.warning(
                f"Couldn't find the number of quotes asked for. Found {len(deduplicated_quotes)} quotes, but need {self.num_quotes_to_evaluate_from_search} quotes"
            )

        most_relevant_quotes = deduplicated_quotes[
            : self.num_quotes_to_evaluate_from_search
        ]
        logger.info(
            f"Found {len(deduplicated_quotes)} quotes and "
            f"{len(set([quote.source.url for quote in deduplicated_quotes]))} unique urls and "
            f"filtering for top {self.num_quotes_to_evaluate_from_search} quotes"
        )
        return most_relevant_quotes

    async def __get_image_descriptions(
        self, quotes: list[ExaHighlightQuote], prompt: str
    ) -> list[ScreenshotDescription]:
        if self.num_sites_to_screenshot == 0:
            return []

        selected_urls = await self.__decide_on_sources_to_screenshot(
            quotes, prompt
        )

        tasks = [
            self._get_screenshot_descriptions_for_url(url, prompt, quotes)
            for url in selected_urls
        ]
        screenshot_descriptions, _ = (
            run_coroutines_while_removing_and_logging_exceptions(tasks)
        )
        return screenshot_descriptions

    async def _get_screenshot_descriptions_for_url(
        self, url: str, prompt: str, quotes: list[ExaHighlightQuote]
    ) -> ScreenshotDescription:
        url_scraper = UrlScraper()
        image_data = await url_scraper.get_screenshot_of_url_as_base64(url)
        description = await UrlScraper.get_summary_of_screenshot(
            image_data, prompt
        )
        description = description.replace("\n", " ")

        matching_quote = next(
            (quote for quote in quotes if quote.source.url == url),
            None,
        )
        matching_source = matching_quote.source if matching_quote else None
        screenshot_description = ScreenshotDescription(
            image_description=description,
            url=url,
            image_data=image_data,
            title=(matching_source.title if matching_source else None),
            readable_publish_date=(
                matching_source.readable_publish_date
                if matching_source
                else None
            ),
        )
        logger.info(f"Got screenshot description for url {url}")
        return screenshot_description

    async def __decide_on_sources_to_screenshot(
        self, quotes: list[ExaHighlightQuote], prompt: str
    ) -> list[str]:
        if self.num_sites_to_screenshot == 0:
            return []
        # TODO: Make sure that all the urls are unique
        formatted_sources = ""
        for i, quote in enumerate(quotes):
            source = quote.source
            formatted_sources += f"{i}) URL: {source.url} | Title: {source.title} | Published: {source.readable_publish_date}\n"

        website_selection_prompt = clean_indents(
            f"""
            You are a researcher who has been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><>START INSTRUCTIONS<><><><><><>
            {prompt}
            <><><><><><>END INSTRUCTIONS<><><><><><>

            You have already done some initial research and need to select the {self.num_sites_to_screenshot} most relevant sources to do deep dives on.
            You will skim the rest of the sources, and will only be shown the text.
            So it is especially important that you choose any sources that you need to see the visual pictures/images/graphs of if that is needed to follow the instructions.
            If there is a link in the instructions that would be good to follow, you can choose this as well.

            Here are the websites:
            {formatted_sources}

            Walk through your reason step by step then give your final answer.
            Return the urls of the best websites to screenshot as a list of strings like this: ["https://www.website.com/article1", "https://www.placeholder.com"]
            Remember, pick {self.num_sites_to_screenshot} sources. Pick all of them if there are only {self.num_sites_to_screenshot} sources.
            You HAVE to pick {self.num_sites_to_screenshot} sources. Even if you just need one source, please pick more (you may be surprised at what you can find)
            """
        )

        selected_urls = await self.llm.invoke_and_return_verified_type(
            website_selection_prompt, list[str]
        )
        assert (
            len(selected_urls) == self.num_sites_to_screenshot
        ), f"Expected {self.num_sites_to_screenshot} sources to be chosen, but the AI chose {len(selected_urls)}. There were {len(quotes)} quotes and {len(set([quote.source.url for quote in quotes]))} unique urls"
        logger.info(
            f"Chose these sources for deep dives: {[url for url in selected_urls]}"
        )
        return selected_urls

    async def __compile_report(
        self,
        quotes: list[ExaHighlightQuote],
        original_instructions: str,
        image_descriptions: list[ScreenshotDescription],
    ) -> str:
        assert len(quotes) > 0, "No search results found"
        assert (
            len(quotes) <= self.num_quotes_to_evaluate_from_search
        ), "Too many search results found"

        search_result_context = (
            self.__turn_highlights_into_search_context_for_prompt(quotes)
        )
        image_context = self.__turn_screenshots_into_search_context(
            image_descriptions
        )

        logger.info(
            f"Generating response using {len(quotes)} quotes and {len(image_descriptions)} images"
        )
        logger.debug(f"Search results:\n{search_result_context}")
        logger.debug(f"Image descriptions:\n{image_context}")

        prompt = clean_indents(
            f"""
            Today is {datetime.now().strftime("%Y-%m-%d")}.
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><><><><><><><>
            {original_instructions}
            <><><><><><><><><><><><>

            After searching the internet, you found the following results. Results are included between <><><><><><><><><><><><> tags.
            <><><><><><><><><><><><>
            {search_result_context}

            {image_context}
            <><><><><><><><><><><><>

            Please follow the instructions and use the search results to answer the question. Unless the instructions specifify otherwise, please cite your sources inline and use markdown formatting.

            For text sources, cite like this:
            > SpaceX successfully completed a full flight test [1].

            For image sources, cite like this:
            > The webpage shows a detailed diagram [Image 1].
            """
        )
        report = await self.llm.invoke(prompt)
        return report

    @staticmethod
    def __turn_highlights_into_search_context_for_prompt(
        highlights: list[ExaHighlightQuote],
    ) -> str:
        search_context = "Webpage Quotes:\n"
        for i, highlight in enumerate(highlights):
            url = highlight.source.url
            title = highlight.source.title
            publish_date = highlight.source.readable_publish_date
            search_context += f'[{i+1}] "{highlight.highlight_text}". [This quote is from {url} titled "{title}", published on {publish_date}]\n'
        return search_context

    @staticmethod
    def __turn_screenshots_into_search_context(
        screenshots: list[ScreenshotDescription],
    ) -> str:
        if not screenshots:
            return ""

        search_context = "Webpage Screenshots:\n"
        for i, screenshot in enumerate(screenshots):
            url = screenshot.url
            title = screenshot.title
            publish_date = screenshot.readable_publish_date
            search_context += f'[Image {i+1}] "{screenshot.image_description}". [This quote is from {url} titled "{title}", published on {publish_date}]\n'
        return search_context

    def __add_quote_links_to_citations(
        self, report: str, highlights: list[ExaHighlightQuote]
    ) -> str:
        for i, highlight in enumerate(highlights):
            citation_num = i + 1

            less_than_10_words = len(highlight.highlight_text.split()) < 10
            if less_than_10_words:
                text_fragment = highlight.highlight_text
            else:
                first_five_words = " ".join(
                    highlight.highlight_text.split()[:5]
                )
                last_five_words = " ".join(
                    highlight.highlight_text.split()[-5:]
                )
                encoded_first_five_words = urllib.parse.quote(
                    first_five_words, safe=""
                )
                encoded_last_five_words = urllib.parse.quote(
                    last_five_words, safe=""
                )
                text_fragment = (
                    f"{encoded_first_five_words},{encoded_last_five_words}"
                )
            text_fragment = text_fragment.replace("(", "%28").replace(
                ")", "%29"
            )
            text_fragment = text_fragment.replace("-", "%2D").strip(",")
            fragment_url = f"{highlight.source.url}#:~:text={text_fragment}"

            report = self.__replace_citation_number_with_url_markdown(
                report, citation_num, fragment_url, is_image=False
            )
        return report

    def __add_image_links_to_citations(
        self, report: str, image_descriptions: list[ScreenshotDescription]
    ) -> str:
        for i in range(len(image_descriptions)):
            image_num = i + 1
            report = self.__replace_citation_number_with_url_markdown(
                report, image_num, image_descriptions[i].url, is_image=True
            )
        return report

    def __replace_citation_number_with_url_markdown(
        self, report: str, citation_num: int, url: str, is_image: bool
    ) -> str:
        citation_text = (
            f"Image {citation_num}" if is_image else str(citation_num)
        )
        if self.use_citation_brackets:
            markdown_url = f"\\[[{citation_text}]({url})\\]"
        else:
            markdown_url = f"[{citation_text}]({url})"

        pattern = re.compile(
            r"(?:\\\[)?(\[{}\](?:\(.*?\))?)(?:\\\])?".format(citation_text)
        )
        new_report = pattern.sub(markdown_url, report)
        return new_report

    @staticmethod
    def _get_cheap_input_for_invoke() -> str:
        return "What is the recent news on SpaceX?"

    @staticmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input() -> str:
        return "Mock Report: Pretend this is an extensive report"
