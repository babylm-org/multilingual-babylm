"""
Ririro resource fetcher for multilingual BabyLM datasets.
Returns data in a format compatible with DocumentConfig.
"""

import os
import time
import base64
import hashlib
import cloudscraper

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag

try:
    from iso639 import Lang  # type: ignore
except Exception:
    Lang = None  # type: ignore

from openai import OpenAI
from pydantic import BaseModel
from multilingual_res.base import BaseResourceFetcher
from typing import Optional, List, Dict

RIRIRO_LANGS = {
    "en": {"url": "https://ririro.com/", "script": "Latn"},
    "nl": {"url": "https://ririro.com/nl/", "script": "Latn"},
    "es": {"url": "https://ririro.com/es/", "script": "Latn"},
    "de": {"url": "https://ririro.com/de/", "script": "Latn"},
    "fr": {"url": "https://ririro.com/fr/", "script": "Latn"},
    "pl": {"url": "https://ririro.com/pl/", "script": "Latn"},
    "sr": {"url": "https://ririro.com/sr/", "script": "Cyrl"},
    "it": {"url": "https://ririro.com/it/", "script": "Latn"},
    "id": {"url": "https://ririro.com/id/", "script": "Latn"},
    "ar": {"url": "https://ririro.com/ar/", "script": "Arab"},
    "pt": {"url": "https://ririro.com/pt/", "script": "Latn"},
    "hi": {"url": "https://ririro.com/hi/", "script": "Deva"},
    "tr": {"url": "https://ririro.com/tr/", "script": "Latn"},
    "uk": {"url": "https://ririro.com/uk/", "script": "Cyrl"},
    "fa": {"url": "https://ririro.com/fa/", "script": "Arab"},
    "ru": {"url": "https://ririro.com/ru/", "script": "Cyrl"},
    "so": {"url": "https://ririro.com/so/", "script": "Latn"},
    "bg": {"url": "https://ririro.com/bg/", "script": "Cyrl"},
}


class AgeEstimate(BaseModel):
    age: int


class RiriroFetcher(BaseResourceFetcher):
    def __init__(self):
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

        self.scraper = cloudscraper.create_scraper()

        self.client = None
        self.llm_mode = None
        if GEMINI_API_KEY:
            self.client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
            self.llm_mode = "gemini"
        elif OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.llm_mode = "openai"

    def _get_book_links(self, main_url):
        resp = self.scraper.get(main_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for article in soup.find_all("article", class_="post-item"):
            if not isinstance(article, Tag):
                continue
            entry = article.find("div", class_="entry-details")
            if not entry or not isinstance(entry, Tag):
                continue
            h2 = entry.find("h2", class_="entry-title")
            if not h2 or not isinstance(h2, Tag):
                continue
            a = h2.find("a")
            href = a["href"] if a and isinstance(a, Tag) and "href" in a.attrs else None
            img_src = None
            if href:
                a_img = None
                for a_candidate in article.find_all("a"):
                    if not isinstance(a_candidate, Tag):
                        continue
                    candidate_href = (
                        a_candidate["href"] if "href" in a_candidate.attrs else None
                    )
                    if candidate_href and candidate_href == href:
                        a_img = a_candidate
                        break
                if a_img:
                    img = None
                    imgs = a_img.find_all("img")
                    for img_candidate in imgs:
                        if isinstance(img_candidate, Tag):
                            img = img_candidate
                            break
                    if img and "src" in img.attrs:
                        src = str(img["src"])
                        if src.startswith("/"):
                            img_src = "https://ririro.com" + src
                        else:
                            img_src = src
            if href:
                links.append((urljoin(main_url, str(href)), img_src))
        return links

    def _extract_book_content(self, book_url):
        resp = self.scraper.get(book_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        h2_read = soup.find("h2", id="read")
        content = []
        if h2_read:
            for sib in h2_read.find_all_next():
                if getattr(sib, "name", None) == "h2" and sib is not h2_read:
                    break
                if getattr(sib, "name", None) == "p":
                    text = sib.get_text(strip=True)
                    if text:
                        content.append(text)
        return "\n\n".join(content) if content else None

    def _download_image_to_base64(self, url):
        resp = self.scraper.get(url)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")

    def _get_lower_age_from_image_b64(self, base64_image):
        if self.client is None:
            return None
        if self.llm_mode == "gemini":
            model = "gemini-2.5-flash"
        elif self.llm_mode == "openai":
            model = "gpt-4.1-nano"
        else:
            return None

        prompt = "What is the minimum age shown in this image? Only answer with a single integer as the field 'age', e.g. {\"age\": 3}."
        completion = self.client.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the minimum age from the image and return as a JSON object with an integer field 'age'.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            response_format=AgeEstimate,
        )
        parsed = completion.choices[0].message.parsed
        if parsed and hasattr(parsed, "age"):
            return parsed.age
        return None

    def fetch(
        self, language_code: str, script_code: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch Ririro data for a given language code and optional script code.
        Returns a list of dicts with keys: text, doc-id, metadata (for DocumentConfig)
        """
        # Normalize language code to ISO-639-1 if needed
        lang_key = language_code
        if lang_key not in RIRIRO_LANGS or len(lang_key) != 2:
            try:
                lang_key = (
                    Lang(language_code).pt1 if Lang is not None else language_code
                )
            except Exception:
                print(f"Ririro not available for language: {language_code}")
                return []
        lang_info = RIRIRO_LANGS.get(lang_key)
        if not lang_info:
            print(f"Ririro not available for language: {language_code}")
            return []
        main_url = lang_info["url"]
        script = script_code if script_code else lang_info["script"]
        book_links = self._get_book_links(main_url)
        results = []
        for i, (url, age_img_url) in enumerate(book_links, 1):
            try:
                content = self._extract_book_content(url)
                if not content:
                    continue
                title = urlparse(url).path.strip("/").split("/")[-1]
                doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
                # Age estimate logic
                if age_img_url and self.llm_mode:
                    b64img = self._download_image_to_base64(age_img_url)
                    lower_age = self._get_lower_age_from_image_b64(b64img)
                    if lower_age and 2 <= lower_age <= 12:
                        age_estimate = f"{lower_age}-12"
                    else:
                        age_estimate = "n/a"
                else:
                    age_estimate = "n/a"
                metadata = {
                    "category": "child-books",
                    "data-source": "Ririro",
                    "script": script,
                    "age-estimate": age_estimate,
                    "license": "cc-by-nc-4.0",
                    "misc": {
                        "source_url": url,
                        "title": title,
                        "multilingual_resource": "ririro",
                    },
                }
                results.append(
                    {"text": content, "doc-id": doc_id, "metadata": metadata}
                )
            except Exception as e:
                print(f"  Error fetching {url}: {e}")
            time.sleep(1)
        print(
            f"Fetched {len(results)} documents from Ririro for language '{language_code}'"
        )
        return results
