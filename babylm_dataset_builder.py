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
        self.validate_misc()

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
        if self.category not in allowed_categories and not self.category.startswith(
            "padding-"
        ):
            raise ValueError(
                f"Category '{self.category}' must be one of: {allowed_categories} or start with 'padding-'"
            )

    def validate_script(self):
        """Validate that script is a valid ISO 15924 code."""
        if not validate_script_code(self.script):
            if not validate_script_code(get_script_code_by_name(self.script)):
                raise ValueError(
                    f"Invalid script code '{self.script}'."
                    " Please use a valid ISO 15924 script code (e.g., Latn, Cyrl, Arab, etc.)"
                )

    def validate_misc(self):
        # convert to string
        if isinstance(self.misc, dict):
            self.misc = json.dumps(self.misc)
        else:
            self.misc = str(self.misc)

        # check valid JSON string
        try:
            json.loads(self.misc)
        except json.JSONDecodeError:
            # fix if just empty
            if self.misc.strip() == "":
                self.misc = "{}"
            else:
                raise ValueError(
                    f"Misc field must be a valid JSON string. Got: {self.misc}"
                )


@dataclass
class DatasetConfig:
    """Configuration for a BabyLM dataset."""

    language_code: str  # ISO 639-3 code

    @property
    def dataset_name(self) -> str:
        return f"babylm-{self.language_code}"


class BabyLMDatasetBuilder:
    @staticmethod
    def bytes_in_text(text: str) -> float:
        """
        Calculate the size of text in MB (UTF-8 encoded).
        """
        return len(text.encode("utf-8")) / 1_000_000

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
                # Backward compatibility: normalize to 'doc-id'
                existing_df = self.normalize_identifier_column(existing_df)
                self._existing_doc_ids = set(existing_df["doc-id"].astype(str))
                self._existing_documents = existing_df.to_dict(orient="records")
            else:
                print("Existing dataset not found for merge.")

    def calculate_dataset_size_mb(self) -> float:
        """
        Calculate the total dataset size in MB by summing the size of each text row.
        """
        if self.dataset_table is None or "text" not in self.dataset_table:
            return 0.0
        return self.dataset_table["text"].apply(self.bytes_in_text).sum()

    def add_document(
        self,
        text: str,
        document_id: str,
        document_config: DocumentConfig,
        additional_metadata: Optional[dict] = None,
    ) -> None:
        """Add a document to the dataset with its own configuration."""
        # Avoid duplicates by doc-id
        if hasattr(self, "_existing_doc_ids") and document_id in self._existing_doc_ids:
            return  # Skip duplicate
        # Create document metadata
        document = {
            "doc-id": document_id,
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
        document_config_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
    Add multiple documents from an iterable of dicts with 'text', 'doc-id' (or legacy 'doc_id'), and 'metadata'.

        Args:
            documents: List of dicts with keys 'text', 'doc-id' (or 'doc_id'), and 'metadata' (optional)
            default_document_config: Default DocumentConfig for documents
        """
        for doc in documents:
            text = doc["text"]
            document_id = doc.get("doc-id") or doc.get("doc_id")
            if document_id is None:
                # Compute from text for robustness
                document_id = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            metadata = doc.get("metadata", {})

            # Prepare misc, merging with existing misc if present
            base_params = document_config_params or {}
            misc = metadata.get("misc", base_params.get("misc"))
            if misc is None:
                misc = {}
            # Add source_url and source_identifier to misc if present
            if "source_url" in metadata:
                misc = dict(misc)  # ensure it's a dict copy
                misc["source_url"] = metadata["source_url"]
            if "source_identifier" in metadata:
                misc = dict(misc)
                misc["source_identifier"] = metadata["source_identifier"]

            try:
                category = metadata.get("category") or base_params.get("category")
                data_source = metadata.get("data-source") or base_params.get("data-source")
                script = metadata.get("script") or base_params.get("script")
                age_estimate = metadata.get("age-estimate") or base_params.get("age-estimate")
                license_val = metadata.get("license") or base_params.get("license")
                if (
                    category is None
                    or data_source is None
                    or script is None
                    or age_estimate is None
                    or license_val is None
                ):
                    raise ValueError(
                        "Missing required document configuration fields: category, data-source, script, age-estimate, license"
                    )
                # Type narrow for static analysis
                assert isinstance(category, str)
                assert isinstance(data_source, str)
                assert isinstance(script, str)
                assert isinstance(age_estimate, str)
                assert isinstance(license_val, str)
                misc_str = json.dumps(misc) if isinstance(misc, dict) else str(misc)
                doc_config = DocumentConfig(
                    category=category,
                    data_source=data_source,
                    script=script,
                    age_estimate=age_estimate,
                    license=license_val,
                    misc=misc_str,
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
        Backward-compat shim: ensure an identifier column exists.
        Prefer 'doc-id'; if only 'doc_id' exists, rename it; if neither, compute from text.
        """
        return self.normalize_identifier_column(df)

    def normalize_identifier_column(self, df: pd.DataFrame) -> pd.DataFrame:
        # If legacy name present, rename
        if "doc-id" in df.columns:
            return df
        if "doc_id" in df.columns:
            return df.rename(columns={"doc_id": "doc-id"})
        # Else compute from text
        print(
            "Identifier column missing. Generating 'doc-id' from text for each record."
        )
        df["doc-id"] = df["text"].apply(
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
            document_id = doc["doc-id"]
            document_config = doc["document_config"]

            row = {
                "text": text,
                "doc-id": document_id,
                "category": document_config.category,
                "data-source": document_config.data_source,
                "script": document_config.script,
                "age-estimate": document_config.age_estimate,
                "license": document_config.license,
                "misc": document_config.misc,
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure identifier column
        df = self.normalize_identifier_column(df)
        # Remove duplicates by doc-id (keep first occurrence)
        df = df.drop_duplicates(subset=["doc-id"])
        # Ensure all values in 'text' column are strings
        df["text"] = df["text"].astype(str)
        # Filter rows where 'text' is not empty after stripping whitespace
        df = df[df["text"].str.strip().str.len() > 0]
        self.dataset_table = df
        return df

    def save_dataset(self) -> None:
        # Save as CSV and parquet for flexibility
        if self.dataset_table is None:
            raise ValueError(
                "dataset_table is None. Call create_dataset_table() before save_dataset()."
            )
        csv_path = self.output_dir / f"{self.dataset_config.dataset_name}_dataset.csv"
        parquet_path = (
            self.output_dir / f"{self.dataset_config.dataset_name}_dataset.parquet"
        )

        self.dataset_table.to_csv(csv_path, index=False)
        self.dataset_table.to_parquet(parquet_path, index=False)

        print("Dataset table saved to:")
        print(f"  - {csv_path}")
        print(f"  - {parquet_path}")

        # Print dataset size in MB
        size_mb = self.calculate_dataset_size_mb()
        print(f"Final dataset size: {size_mb:.2f} MB (UTF-8 bytes)")

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
        if self.dataset_table is None:
            # Attempt to build from current documents
            self.create_dataset_table()
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
