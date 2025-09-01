"""Resource fetcher for GlotStoryBook datasets from HuggingFace."""

from collections import defaultdict
import hashlib
import os
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from multilingual_res.base import BaseResourceFetcher
from typing import List, Dict, cast


class GlotStorybookFetcher(BaseResourceFetcher):
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            load_dotenv()
            self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable not set. Please set it or add it to your .env file."
            )

    def fetch(self, language_code: str) -> List[Dict]:
        """
        Fetch GlotStoryBook data for a given language code and optional script code.
        Returns a list of dicts with keys: text, doc-id, metadata (for DocumentConfig)
        """
        dataset = load_dataset("cis-lmu/GlotStoryBook", split="train")
        grouped = defaultdict(list)
        for row in dataset:
            d = cast(dict, row)
            grouped[d["File Name"]].append(d)
        new_rows = []
        for rows in grouped.values():
            sorted_rows = sorted(rows, key=lambda x: x["Text Number"])
            full_text = " ".join(r["Text"] for r in sorted_rows if r.get("Text"))
            merged_row = {
                key: sorted_rows[0][key]
                for key in sorted_rows[0]
                if key not in ["Text", "Text Number"]
            }
            merged_row["Text"] = full_text
            new_rows.append(merged_row)
        dataset = Dataset.from_list(new_rows)

        # Filter by language using Dataset filter
        filtered = dataset.filter(lambda x: x["ISO639-3"] == language_code)
        results = []
        for doc in filtered:
            d = cast(dict, doc)
            text = d.get("Text")
            if text:
                script = d.get("Script")
                data_license = d.get("License")
                author = d.get("Text By")
                translator = d.get("Translation By")
                data_source = "GlotStoryBook"
                description = "Children StoryBooks for 180 languages."
                source_identifier = d.get("Source")
                misc = {
                    "translator": translator,
                    "author": author,
                    "description": description,
                    "source_identifier": source_identifier,
                    "multilingual_resource": "glotstorybook",
                }
                metadata = {
                    "category": "child-books",
                    "data-source": data_source,
                    "script": script,
                    "age-estimate": "n/a",
                    "license": data_license,
                    "misc": misc,
                }
                doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                results.append({"text": text, "doc-id": doc_id, "metadata": metadata})
        print(
            f"Fetched {len(results)} documents from GlotStoryBook for language '{language_code}'"
        )
        return results
