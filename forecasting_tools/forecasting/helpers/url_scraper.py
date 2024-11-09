import base64
import io
import logging

from crawl4ai import AsyncWebCrawler
from PIL import Image

logger = logging.getLogger(__name__)
Base64Image = str


class UrlScraper:

    async def get_screenshot_of_url_as_base64(self, url: str) -> Base64Image:
        cropped_image = await self.get_screenshot_of_url_as_file(url)
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        return base64_image

    async def get_screenshot_of_url_as_file(self, url: str) -> Image.Image:

        async with AsyncWebCrawler(
            verbose=False,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        ) as crawler:
            time_to_load = 15
            result = await crawler.arun(
                url=url,
                screenshot=True,  # Enable screenshot
                screenshot_wait_for=time_to_load,  # Wait before capture
                magic=True,  # Enable all anti-detection features
                process_iframes=True,  # Process iframe content
                remove_overlay_elements=True,  # Remove popups/modals
                page_timeout=60000,
                bypass_cache=True,
                delay_before_return_html=time_to_load,  # Wait for images to load
                verbose=False,
            )

            if not result.screenshot:
                raise RuntimeError(
                    f"Failed to get screenshot of {url}. Error: {result.error_message}"
                )

            image_data = base64.b64decode(result.screenshot)
            image = Image.open(io.BytesIO(image_data))

            target_height = 1500
            cropped_image = image.crop(
                (0, 0, image.width, min(image.height, target_height))
            )
            return cropped_image
