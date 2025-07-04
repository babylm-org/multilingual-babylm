"""
General dataset builder for BabyLM multilingual datasets.
This can be used for any data source, not just OpenSubtitles.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset, load_from_disk

from language_scripts import validate_script_code


@dataclass
class DocumentConfig:
    """Configuration for a single document in the BabyLM dataset."""

    category: str  # e.g., child-directed-speech, educational, child-books, etc.
    data_source: str
    script: str  # ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)
    age_estimate: str  # Specific age, range, or "n/a"
    license: str  # cc-by, cc-by-sa, etc.
    misc: Optional[dict] = None

    def __post_init__(self):
        """Post-initialization validation."""
        self.validate_category()
        self.validate_script()

    def validate_category(self):
        """Validate that category is one of the allowed values."""
        allowed_categories = {
            "child-directed-speech",
            "educational",
            "child-books",
            "child-wiki",
            "child-news",
            "subtitles",
            "qed",
            "child-available-speech",
            "simplified-text",
        }
        if self.category not in allowed_categories:
            raise ValueError(
                f"Category '{self.category}' must be one of: {allowed_categories}"
            )

    def validate_script(self):
        """Validate that script is a valid ISO 15924 code."""
        if not validate_script_code(self.script):
            raise ValueError(
                f"Invalid script code '{self.script}'."
                " Please use a valid ISO 15924 script code (e.g., Latn, Cyrl, Arab, etc.)"
            )


@dataclass
class DatasetConfig:
    """Configuration for a BabyLM dataset."""

    language_code: str  # ISO 639-3 code

    @property
    def dataset_name(self) -> str:
        return f"babylm-{self.language_code}"


class BabyLMDatasetBuilder:
    """Build and manage BabyLM datasets from various sources."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        output_dir: Path = Path("./babylm_datasets"),
    ):
        self.dataset_config = dataset_config

        self.output_dir = output_dir / self.dataset_config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "dataset_metadata.json"

        self.documents: list[dict] = []

    def add_document(
        self,
        text: str,
        document_id: str,
        document_config: DocumentConfig,
        additional_metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the dataset with its own configuration."""
        # Create document metadata
        document = {
            "document_id": document_id,
            "document_config": document_config,
            "text": text,  # Store the text content
        }
        # Add optional fields
        if document_config.misc:
            document["misc"] = document_config.misc
        if additional_metadata:
            document["additional_metadata"] = additional_metadata

        self.documents.append(document)

    def add_documents_from_iterable(
        self,
        documents,
        default_document_config: DocumentConfig,
    ) -> None:
        """
        Add multiple documents from an iterable of dicts with 'text', 'doc_id', and 'metadata'.

        Args:
            documents: List of dicts with keys 'text', 'doc_id', and 'metadata' (optional)
            default_document_config: Default DocumentConfig for documents
        """
        for doc in documents:
            text = doc["text"]
            document_id = doc["doc_id"]
            metadata = doc.get("metadata", {})

            # Prepare misc, merging with existing misc if present
            misc = metadata.get("misc", default_document_config.misc)
            if misc is None:
                misc = {}
            # Add source_url and source_identifier to misc if present
            if "source_url" in metadata:
                misc = dict(misc)  # ensure it's a dict copy
                misc["source_url"] = metadata["source_url"]
            if "source_identifier" in metadata:
                misc = dict(misc)
                misc["source_identifier"] = metadata["source_identifier"]
            doc_config = DocumentConfig(
                category=metadata.get("category") or default_document_config.category,
                data_source=metadata.get("data_source")
                or default_document_config.data_source,
                script=metadata.get("script") or default_document_config.script,
                age_estimate=metadata.get("age_estimate")
                or default_document_config.age_estimate,
                license=metadata.get("license") or default_document_config.license,
                misc=misc,
            )
            config_keys = {
                "category",
                "data_source",
                "script",
                "age_estimate",
                "license",
                "misc",
            }
            additional_metadata = {
                k: v for k, v in metadata.items() if k not in config_keys
            }
            self.add_document(text, document_id, doc_config, additional_metadata)

    def create_dataset_table(self) -> pd.DataFrame:
        """Create the standardized BabyLM dataset table."""
        rows = []
        for doc in self.documents:
            # Read the text
            text = doc["text"]
            document_id = doc["document_id"]
            document_config = doc["document_config"]

            row = {
                "text": text,
                "document_id": document_id,
                "category": document_config.category,
                "data-source": document_config.data_source,
                "script": document_config.script,
                "age-estimate": document_config.age_estimate,
                "license": document_config.license,
            }
            # Add misc field if present
            if doc.get("misc"):
                row["misc"] = json.dumps(doc["misc"])
            else:
                row["misc"] = None
            rows.append(row)

        df = pd.DataFrame(rows)
        # save metadata
        self.dataset_table = df
        return df

    def save_dataset(self) -> None:
        # Save as CSV and parquet for flexibility
        csv_path = self.output_dir / f"{self.dataset_config.dataset_name}_dataset.csv"
        parquet_path = (
            self.output_dir / f"{self.dataset_config.dataset_name}_dataset.parquet"
        )

        self.dataset_table.to_csv(csv_path, index=False)
        self.dataset_table.to_parquet(parquet_path, index=False)

        print("Dataset table saved to:")
        print(f"  - {csv_path}")
        print(f"  - {parquet_path}")

        metadata = {
            "dataset_name": self.dataset_config.dataset_name,
            "language_code": self.dataset_config.language_code,
            "creation_date": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "config": asdict(self.dataset_config),
        }
        self.metadata = metadata

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"Metadata saved to {self.metadata_path}")

    def get_upload_ready_dataset(self) -> dict:
        """
        Get the dataset in a format ready for upload to HuggingFace.

        Returns:
            Dict with 'data' (DataFrame) and 'metadata' (Dict)

        """
        return {"data": self.dataset_table, "metadata": self.metadata}
