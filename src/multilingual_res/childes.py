"""Resource fetcher for CHILDES datasets from HuggingFace."""

import hashlib
import os
from datasets import load_dataset
from dotenv import load_dotenv
from multilingual_res.base import BaseResourceFetcher
from typing import List, Dict, Optional, cast


class ChildesFetcher(BaseResourceFetcher):
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
        Fetch CHILDES data for a given language code and script code.
        Returns a list of dicts with keys: text, doc-id, metadata (for DocumentConfig)
        """
        try:
            dataset = load_dataset(
                "BabyLM-community/formatted-CHILDES", language_code, split="train"
            )
        except ValueError:
            print(f"CHILDES not available for language: {language_code}")
            return []

        results = []
        for doc in dataset:
            d = cast(dict, doc)
            text = d.get("text")
            if text:
                metadata = {
                    "category": d.get("category", "child-directed-speech"),
                    "data-source": d.get("data-source", "unknown"),
                    "script": d.get("script", script_code),
                    "age-estimate": d.get("age-estimate", "n/a"),
                    "license": d.get("license", "unknown"),
                    "misc": {"multilingual_resource": "childes"},
                }
                doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                results.append({"text": text, "doc-id": doc_id, "metadata": metadata})
        print(
            f"Fetched {len(results)} documents from CHILDES for language '{language_code}'"
        )
        return results
