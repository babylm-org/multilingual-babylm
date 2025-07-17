"""Resource fetcher for ChildWiki datasets from HuggingFace."""

import hashlib
import os
from datasets import load_dataset
from dotenv import load_dotenv
from multilingual_res.base import BaseResourceFetcher
from typing import List, Dict, Optional


class ChildWikiFetcher(BaseResourceFetcher):
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            load_dotenv()
            self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable not set. Please set it or add it to your .env file."
            )

    def fetch(
        self, language_code: str, script_code: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch ChildWiki data for a given language code and script code.
        Returns a list of dicts with keys: text, doc_id, metadata (for DocumentConfig)
        """
        dataset = load_dataset("BabyLM-community/baby-wikis", split="train")
        filtered = dataset.filter(lambda x: x["lang"] == language_code)
        results = []
        for doc in filtered:
            text = doc["content"]
            if text is not None:
                data_source = doc["wiki"]
                title = doc["title"]
                metadata = {
                    "category": "child-wiki",
                    "data-source": f"ChildWiki - {data_source}",
                    "script": script_code,
                    "age-estimate": "n/a",
                    "license": "cc-by-sa",
                    "misc": {"title": title},
                }
                doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                results.append(
                    {
                        "text": text,
                        "doc_id": doc_id,
                        "metadata": metadata,
                    }
                )
        print(
            f"Fetched {len(results)} documents from ChildWiki for language '{language_code}'"
        )
        return results
