"""
Resource manager for multilingual_res package.
Allows fetching from any supported resource by name.
"""

from multilingual_res.ririro import RiriroFetcher
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
    else:
        raise ValueError(f"Resource '{resource_name}' not supported.")

    return fetcher.fetch(language_code, script_code)
