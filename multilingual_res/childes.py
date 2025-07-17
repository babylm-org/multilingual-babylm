"""Resource fetcher for CHILDES datasets from HuggingFace."""

import hashlib
import os
from datasets import load_dataset
from dotenv import load_dotenv
from multilingual_res.base import BaseResourceFetcher
from typing import List, Dict, Optional


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
        Returns a list of dicts with keys: text, doc_id, metadata (for DocumentConfig)
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
            text = doc["text"]
            if text is not None:
                metadata = {
                    "category": doc.get("category", "child-directed-speech"),
                    "data-source": doc.get("data-source", "CHILDES"),
                    "script": doc.get("script", script_code),
                    "age-estimate": doc.get("age-estimate", "n/a"),
                    "license": doc.get("license", "unknown"),
                    "misc": "",
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
            f"Fetched {len(results)} documents from CHILDES for language '{language_code}'"
        )
        return results
