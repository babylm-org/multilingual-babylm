"""
General dataset builder for BabyLM multilingual datasets.
This can be used for any data source, not just OpenSubtitles.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import hashlib
import pandas as pd
from language_scripts import validate_script_code, get_script_code_by_name


@dataclass
class DocumentConfig:
    """Configuration for a single document in the BabyLM dataset."""

    category: str  # e.g., child-directed-speech, educational, child-books, etc.
    data_source: str
    script: str  # ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)
    age_estimate: str  # Specific age, range, or "n/a"
    license: str  # cc-by, cc-by-sa, etc.
    misc: str

    def __post_init__(self):
        """Post-initialization validation."""
        self.validate_category()
        self.validate_script()

        # check for None values
        if self.license is None:
            raise ValueError("License must be specified.")
        if self.data_source is None:
            raise ValueError("Data source must be specified.")
        if self.age_estimate is None:
            raise ValueError("Age estimate must be specified.")

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
            "padding",
        }
        if self.category not in allowed_categories:
            raise ValueError(
                f"Category '{self.category}' must be one of: {allowed_categories}"
            )

    def validate_script(self):
        """Validate that script is a valid ISO 15924 code."""
        if not validate_script_code(self.script):
            if not validate_script_code(get_script_code_by_name(self.script)):
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
        merge_existing: bool = True,  # Merge with existing data by default
    ):
        self.dataset_config = dataset_config
        self.output_dir = output_dir / self.dataset_config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "dataset_metadata.json"
        self.documents: list[dict] = []
        self.dataset_table = None

        # Load existing data if present and merge later
        self._existing_doc_ids = set()
        self._existing_documents = []
        if merge_existing:
            csv_path = (
                self.output_dir / f"{self.dataset_config.dataset_name}_dataset.csv"
            )
            parquet_path = (
                self.output_dir / f"{self.dataset_config.dataset_name}_dataset.parquet"
            )
            existing_df = None
            if csv_path.exists():
                try:
                    existing_df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Warning: Could not load existing CSV: {e}")
            elif parquet_path.exists():
                try:
                    existing_df = pd.read_parquet(parquet_path)
                except Exception as e:
                    print(f"Warning: Could not load existing Parquet: {e}")
            if existing_df is not None:
                print(f"Loaded existing data with {len(existing_df)} documents.")
                # Backward compatibility: add doc_id if missing
                if "doc_id" not in existing_df.columns:
                    existing_df = self.add_missing_doc_id(existing_df)
                self._existing_doc_ids = set(existing_df["doc_id"].astype(str))
                self._existing_documents = existing_df.to_dict(orient="records")
            else:
                print("Existing dataset not found for merge.")

    def add_document(
        self,
        text: str,
        document_id: str,
        document_config: DocumentConfig,
        additional_metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the dataset with its own configuration."""
        # Avoid duplicates by doc_id
        if hasattr(self, "_existing_doc_ids") and document_id in self._existing_doc_ids:
            return  # Skip duplicate
        # Create document metadata
        document = {
            "doc_id": document_id,
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
        document_config_params: dict[str, Any] = None,
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
            misc = metadata.get("misc", document_config_params.get("misc"))
            if misc is None:
                misc = {}
            # Add source_url and source_identifier to misc if present
            if "source_url" in metadata:
                misc = dict(misc)  # ensure it's a dict copy
                misc["source_url"] = metadata["source_url"]
            if "source_identifier" in metadata:
                misc = dict(misc)
                misc["source_identifier"] = metadata["source_identifier"]

            # Ensure misc is a valid string
            if misc is None:
                misc = ""

            try:
                doc_config = DocumentConfig(
                    category=metadata.get("category")
                    or document_config_params.get("category"),
                    data_source=metadata.get("data-source")
                    or document_config_params.get("data-source"),
                    script=metadata.get("script")
                    or document_config_params.get("script"),
                    age_estimate=metadata.get("age-estimate")
                    or document_config_params.get("age-estimate", "n/a"),
                    license=metadata.get("license")
                    or document_config_params.get("license", "n/a"),
                    misc=misc,
                )

            except ValueError as e:
                raise ValueError(
                    f"Error in configuration of document with id {document_id}: {e}"
                )

            config_keys = {
                "category",
                "data-source",
                "script",
                "age-estimate",
                "license",
                "misc",
            }
            additional_metadata = {
                k: v for k, v in metadata.items() if k not in config_keys
            }
            self.add_document(text, document_id, doc_config, additional_metadata)

    def add_missing_doc_id(self, df):
        """
        Add a 'doc_id' column to the DataFrame if it doesn't exist.
        Generates SHA256 hash of the 'text' column for each row.
        """
        print(
            "doc_id column missing in existing data. Generating doc_id for each record."
        )
        df["doc_id"] = df["text"].apply(
            lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
        )
        return df

    def create_dataset_table(self) -> pd.DataFrame:
        """Create the standardized BabyLM dataset table."""
        rows = []
        # Add existing documents first (if any)
        if hasattr(self, "_existing_documents") and self._existing_documents:
            rows.extend(self._existing_documents)
        # Add new documents
        for doc in self.documents:
            # Read the text
            text = doc["text"]
            document_id = doc["doc_id"]
            document_config = doc["document_config"]

            row = {
                "text": text,
                "doc_id": document_id,
                "category": document_config.category,
                "data-source": document_config.data_source,
                "script": document_config.script,
                "age-estimate": document_config.age_estimate,
                "license": document_config.license,
                "num_tokens": len(text.split()),
            }
            # Add misc field if present
            if doc.get("misc"):
                row["misc"] = json.dumps(doc["misc"])
            else:
                row["misc"] = ""

            rows.append(row)
        df = pd.DataFrame(rows)
        # Remove duplicates by doc_id (keep first occurrence)
        df = df.drop_duplicates(subset=["doc_id"])
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
            "num_documents": len(self.dataset_table),
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

    def deduplicate_by_text(self) -> None:
        """
        Remove exact duplicate documents based on the hash of the text field.
        Keeps the first occurrence of each unique text.
        Updates self.dataset_table in place.
        """

        if self.dataset_table is None or "text" not in self.dataset_table:
            print("No dataset_table or 'text' column to deduplicate.")
            return
        before = len(self.dataset_table)
        # Compute hash for each text
        self.dataset_table["text_hash"] = self.dataset_table["text"].apply(
            lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
        )
        # Drop duplicates by text_hash
        self.dataset_table = self.dataset_table.drop_duplicates(subset=["text_hash"])
        self.dataset_table = self.dataset_table.drop(columns=["text_hash"])
        after = len(self.dataset_table)
        print(
            f"Deduplicated dataset: removed {before - after} duplicates, {after} remain."
        )
