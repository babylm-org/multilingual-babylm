"""
General dataset builder for BabyLM multilingual datasets.
This can be used for any data source, not just OpenSubtitles.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_from_disk

from language_scripts import validate_script_code


@dataclass
class DocumentConfig:
    """Configuration for a single document in the BabyLM dataset."""

    category: str  # One of the predefined categories
    data_source: str
    script: str  # latin, cyrillic, etc.
    age_estimate: str  # Specific age, range, or "n/a"
    license: str  # cc-by, cc-by-sa, etc.
    misc: dict[str, str] | None = None

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
        additional_metadata: dict | None = None,
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

    def add_documents_from_path(
        self,
        load_path: Path,
        load_format: str,
        default_config_params: dict[str, dict],
        hf_dataset_split: str | None = None,
    ) -> None:
        # Determine file type based on extension

        # load json
        if load_format == "json":
            with open(load_path, encoding="utf-8") as f:
                data = json.load(f)

        # load jsonl
        elif load_format == "jsonl":
            data = []
            with open(load_path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        # load csv
        elif load_format == "csv":
            df = pd.read_csv(load_path)
            data = df.to_dict(orient="records")

        # load from local or HuggingFace dataset
        elif load_format == "hf":
            if load_path.exists():
                # If the path exists, assume it's a local dataset
                ds = load_from_disk(load_path, split=hf_dataset_split)
            else:
                # Otherwise, assume it's a HuggingFace dataset ID
                ds = load_dataset(load_path, split=hf_dataset_split)

            # If no split, ds is a dict of splits, use the first split
            if isinstance(ds, dict):
                ds = next(iter(ds.values()))
                data = ds.to_dict(orient="records")

        for document_id, doc_data in enumerate(data):
            text = doc_data["text"]
            # Create document configuration using default parameters and overrides
            doc_config = {**default_config_params, **doc_data}
            doc_config = DocumentConfig(
                category=doc_data.get("category"),
                data_source=doc_data.get("data_source"),
                script=doc_data.get("script"),
                age_estimate=doc_data.get("age_estimate"),
                license=doc_data.get("license"),
                misc=doc_data.get("misc"),
            )
            additional_metadata = {
                k: v
                for k, v in doc_data.items()
                if k
                not in [
                    "category",
                    "data_source",
                    "script",
                    "age_estimate",
                    "license",
                    "misc",
                    "text",  # ignore text field as well
                ]
            }
            self.add_document(text, document_id, doc_config, additional_metadata)

    def add_documents_from_text_directory(
        self,
        load_path: Path,
        default_config_params: dict[str, dict],
        metadata_mapping: dict[str, dict] | None = None,
    ) -> None:
        """
        Add multiple documents from a directory of text files.

        Args:
            load_path: Directory containing text files
            default_config_params: Default configuration for documents
            metadata_mapping: Optional dict mapping document_id to metadata overrides

        """
        if metadata_mapping is None:
            metadata_mapping = {}

        for text_file in load_path.glob("*.txt"):
            with open(text_file, encoding="utf-8") as f:
                text = f.read()
            document_id = text_file.stem
            doc_metadata = metadata_mapping.get(document_id, {})

            # load metadata for this document
            # doc_metadata overrides the default config for any overlapping keys
            config = {**default_config_params, **doc_metadata}
            doc_config = DocumentConfig(
                category=config.get("category"),
                data_source=config.get("data_source"),
                script=config.get("script"),
                age_estimate=config.get("age_estimate"),
                license=config.get("license"),
                misc=config.get("misc"),
            )
            # Extract additional metadata (non-config fields)
            additional_metadata = {
                k: v
                for k, v in doc_metadata.items()
                if k
                not in [
                    "category",
                    "data_source",
                    "script",
                    "age_estimate",
                    "license",
                    "misc",
                ]
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

    def get_upload_ready_dataset(self) -> dict[str, pd.DataFrame | dict]:
        """
        Get the dataset in a format ready for upload to HuggingFace.

        Returns:
            Dict with 'data' (DataFrame) and 'metadata' (Dict)

        """
        return {"data": self.dataset_table, "metadata": self.metadata}
