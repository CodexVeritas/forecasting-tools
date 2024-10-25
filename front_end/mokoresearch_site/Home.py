import os
import sys

import dotenv
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)


from front_end.mokoresearch_site.app_pages.base_rate_page import BaseRatePage
from front_end.mokoresearch_site.app_pages.benchmark_page import BenchmarkPage
from front_end.mokoresearch_site.app_pages.example_forecasts import ExampleForecastsPage
from front_end.mokoresearch_site.app_pages.forecaster_page import ForecasterPage
from front_end.mokoresearch_site.app_pages.niche_list_researcher_page import NicheListResearchPage
from front_end.mokoresearch_site.app_pages.estimator_page import EstimatorPage
from front_end.mokoresearch_site.app_pages.key_factors_page import KeyFactorsPage
from front_end.mokoresearch_site.helpers.app_page import AppPage
from front_end.mokoresearch_site.helpers.general import footer, header
from src.util.custom_logger import CustomLogger


class HomePage(AppPage):
    FILE_PATH_IN_FRONT_END_FOLDER: str = "Home.py"
    PAGE_DISPLAY_NAME: str = "🏠 Home"
    URL_PATH: str = "/"
    IS_DEFAULT_PAGE: bool = True

    FORECASTER_PAGE: type[AppPage] = ForecasterPage
    BASE_RATE_PAGE: type[AppPage] = BaseRatePage
    NICHE_LIST_RESEARCH_PAGE: type[AppPage] = NicheListResearchPage
    ESTIMATOR_PAGE: type[AppPage] = EstimatorPage
    KEY_FACTORS_PAGE: type[AppPage] = KeyFactorsPage
    EXAMPLE_FORECASTS_PAGE: type[AppPage] = ExampleForecastsPage
    BENCHMARKS_PAGE: type[AppPage] = BenchmarkPage
    NON_HOME_PAGES: list[type[AppPage]] = [
        BASE_RATE_PAGE,
        NICHE_LIST_RESEARCH_PAGE,
        ESTIMATOR_PAGE,
        KEY_FACTORS_PAGE,
        FORECASTER_PAGE,
        EXAMPLE_FORECASTS_PAGE,
        BENCHMARKS_PAGE,
    ]

    @classmethod
    async def async_main(cls) -> None:
        header()
        st.title("What do you want to do?")
        for page in cls.NON_HOME_PAGES:
            label = page.PAGE_DISPLAY_NAME
            if st.button(label, key=label):
                st.switch_page(page.convert_to_streamlit_page())
        footer()


if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    all_pages = [HomePage] + HomePage.NON_HOME_PAGES
    navigation = st.navigation([page.convert_to_streamlit_page() for page in all_pages])
    st.set_page_config(page_title="Moko Research", page_icon=":material/explore:")
    navigation.run()
