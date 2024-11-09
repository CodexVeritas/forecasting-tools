from PIL import Image

from forecasting_tools.forecasting.helpers.url_scraper import UrlScraper


async def test_get_screenshot_as_file() -> None:
    url_scraper = UrlScraper()
    test_url = "https://example.com"

    screenshot_file = await url_scraper.get_screenshot_of_url_as_file(test_url)
    assert isinstance(screenshot_file, Image.Image)
    assert screenshot_file.width > 0
    assert screenshot_file.height > 0
    assert screenshot_file.height <= 1500


async def test_get_screenshot_as_base64() -> None:
    url_scraper = UrlScraper()
    test_url = "https://example.com"

    base64_screenshot = await url_scraper.get_screenshot_of_url_as_base64(
        test_url
    )
    assert isinstance(base64_screenshot, str)
    assert len(base64_screenshot) > 0
    assert base64_screenshot.startswith(
        "/9j/"
    ) or base64_screenshot.startswith("iVBORw0KGgo")
