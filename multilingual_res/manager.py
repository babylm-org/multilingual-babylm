"""
Resource manager for multilingual_res package.
Allows fetching from any supported resource by name.
"""

from multilingual_res.ririro import RiriroFetcher
from multilingual_res.glotstorybook import GlotStorybookFetcher
from multilingual_res.childwiki import ChildWikiFetcher
from typing import Optional


def fetch_resource(
    resource_name: str, language_code: str, script_code: Optional[str] = None
):
    """
    Fetch data from a supported resource.
    Returns a list of dicts (see resource fetcher for format).
    """
    if resource_name == "ririro":
        fetcher = RiriroFetcher()
        return fetcher.fetch(language_code, script_code)
    elif resource_name == "glotstorybook":
        fetcher = GlotStorybookFetcher()
        # GlotStorybookFetcher.fetch does not use script_code
        return fetcher.fetch(language_code)
    elif resource_name == "childwiki":
        fetcher = ChildWikiFetcher()
        return fetcher.fetch(language_code, script_code)
    else:
        raise ValueError(f"Resource '{resource_name}' not supported.")
