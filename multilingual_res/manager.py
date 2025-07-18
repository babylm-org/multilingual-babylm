"""
Resource manager for multilingual_res package.
Allows fetching from any supported resource by name.
"""

from multilingual_res.ririro import RiriroFetcher
from multilingual_res.glotstorybook import GlotStorybookFetcher
from multilingual_res.childwiki import ChildWikiFetcher
from multilingual_res.childes import ChildesFetcher
from typing import Any, Optional
import re
import json


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
    elif resource_name == "childes":
        fetcher = ChildesFetcher()
        return fetcher.fetch(language_code, script_code)
    else:
        raise ValueError(f"Resource '{resource_name}' not supported.")


def remove_resource(resource_name: str, docs: list[dict[str, Any]]) ->  list[dict[str, Any]]:
    # general resource removal method — for updated metadata

    processed_docs = []
    for doc in docs:
        metadata = doc.get("metadata")
        category = metadata.get("category", "n/a")
        data_source = metadata.get("data-source", "n/a")
        age_estimate = metadata.get("age-estimate", "n/a")

        try:
            misc = metadata.get("misc", {})
        except json.JSONDecodeError:
            misc = {}

        # for our purposes it's OK
        if not isinstance(misc, dict):
            misc = {}


        multilingual_resource = misc.get("multilingual_resource", "n/a")
        if multilingual_resource == resource_name:
            continue 

        if resource_name == "ririro" and data_source == "Ririro":
            continue
        elif resource_name == "glotstorybook" and data_source == "GlotStoryBook":
            continue

        elif resource_name == "childwiki":
            wikis = {'vikidia', 'grundschulwiki', 'wikikids', 'mini-klexikon', 'kiwithek', 'klexikon', 'txikipedia', 'wikimini'}
            if category == "child-wiki" and data_source in wikis:
                continue

        elif resource_name == "childes":
            condition = category == "child-directed-speech"
            condition &= (
                data_source.lower() == "childes"
                or re.fullmatch(r"\S+/\d+", data_source) is not None
            )
            condition &= (
                age_estimate == "n/a"
                or ";" in age_estimate
                or age_estimate == "nan"
            )
            if condition:
                continue

        processed_docs.append(doc)

    print(f"\n{'=' * 60}")
    print(f"Removed {len(docs) - len(processed_docs)} documents from {resource_name} resource.")
    print(f"Remaining documents: {len(processed_docs)}")
    print(f"{'=' * 60}\n")

    return processed_docs
