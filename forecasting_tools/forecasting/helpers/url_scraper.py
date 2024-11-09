import base64
import io
import logging
from typing import NewType

from crawl4ai import AsyncWebCrawler
from PIL import Image

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.forecasting.helpers.configured_llms import (
    VisionData,
    VisionLlm,
)
from forecasting_tools.util import file_manipulation

logger = logging.getLogger(__name__)
Base64Image = NewType("Base64Image", str)


class UrlScraper:

    async def get_screenshot_of_url_as_base64(self, url: str) -> Base64Image:
        cropped_image = await self.get_screenshot_of_url_as_image(url)
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        return Base64Image(base64_image)

    async def get_screenshot_of_url_as_image(self, url: str) -> Image.Image:
        initial_wait_time = 15
        fallback_wait_time = 2

        async with AsyncWebCrawler(
            verbose=False,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        ) as crawler:
            for wait_time in [initial_wait_time, fallback_wait_time]:
                screenshot = await self._get_screenshot_with_wait_time(
                    url, wait_time, crawler
                )
                if screenshot:
                    return screenshot
                else:
                    logger.warning(
                        f"Failed to get screenshot of {url} after {wait_time} seconds"
                    )

            raise RuntimeError(
                f"Failed to get screenshot of {url} after multiple attempts"
            )

    async def _get_screenshot_with_wait_time(
        self, url: str, wait_time: int, crawler: AsyncWebCrawler
    ) -> Image.Image | None:
        result = await crawler.arun(
            url=url,
            screenshot=True,
            screenshot_wait_for=wait_time,
            magic=True,
            process_iframes=True,
            remove_overlay_elements=True,
            page_timeout=60000,
            bypass_cache=True,
            delay_before_return_html=wait_time,
            verbose=False,
        )

        if not result.screenshot:
            return None

        image_data = base64.b64decode(result.screenshot)
        image = Image.open(io.BytesIO(image_data))

        target_height = 1250
        cropped_image = image.crop(
            (0, 0, image.width, min(image.height, target_height))
        )
        file_manipulation.write_image_file(
            "logs/latest_screenshot.png", cropped_image
        )
        return cropped_image

    @classmethod
    async def get_summary_of_screenshot(
        cls,
        image_data: Base64Image,
        prompt: str,
    ) -> str:
        vision_llm = VisionLlm()

        vision_prompt = clean_indents(
            f"""
            You have been given the following instructions. Instructions are included between <><><><><><><><><><><><> tags.

            <><><><><><>START INSTRUCTIONS<><><><><><>
            {prompt}
            <><><><><><>END INSTRUCTIONS<><><><><><>

            You are looking at a screenshot of a webpage that may contain relevant information for these instructions.
            Please describe what you see in this webpage screenshot, focusing on the information that will help you to answer the instructions.
            If you are describing the graph, please give specifics about the data displayed
            - Mention a couple of data points you can see throughout the graph
            - Give an assessment of general trends of the data.
            - Please note when you are estimating versus when you can clearly see the value of a point on the graph
            Give a paragraph long description
            """
        )
        summary = await vision_llm.invoke(
            VisionData(
                prompt=vision_prompt,
                b64_image=image_data,
                image_resolution="high",
            )
        )
        return summary
