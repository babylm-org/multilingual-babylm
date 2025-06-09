"""
HuggingFace dataset uploader for BabyLM datasets.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, DatasetDict


class HFDatasetUploader:
    """Handle uploading BabyLM datasets to HuggingFace."""

    def __init__(self, token: Optional[str] = None):
        load_dotenv()
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not found. Set it in .env or pass directly.")

        self.api = HfApi(token=self.token)

    def upload_babylm_dataset(
        self,
        dataset_dir: Path,
        repo_id: str,
        private: bool = True,
        create_dataset_card: bool = True,
        create_repo_if_missing: bool = True,
    ) -> None:
        """
        Upload a BabyLM dataset to HuggingFace.

        Args:
            dataset_dir: Directory containing the dataset files
            repo_id: HuggingFace repo ID (e.g., "username/babylm-eng")
            private: Whether to make the repo private
            create_dataset_card: Whether to create a README
            create_repo_if_missing: Whether to create the repo if it doesn't exist
        """
        # Optionally ensure repository exists
        if create_repo_if_missing:
            try:
                print(f"Checking repository {repo_id}...")
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    token=self.token,
                    exist_ok=True,
                )
                print(f"Repository {repo_id} ready.")
            except Exception as e:
                print(f"Error with repository: {e}")
                return

        # Find dataset files
        parquet_files = list(dataset_dir.glob("*.parquet"))
        csv_files = list(dataset_dir.glob("*.csv"))

        if parquet_files:
            # Prefer parquet format
            data_file = parquet_files[0]
            df = pd.read_parquet(data_file)
        elif csv_files:
            # Fallback to CSV
            data_file = csv_files[0]
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"No dataset files found in {dataset_dir}")

        # Calculate token statistics
        def count_tokens(text):
            if not isinstance(text, str):
                return 0
            return len(text.split())

        df["num_tokens"] = df["text"].apply(count_tokens)
        total_tokens = int(df["num_tokens"].sum())
        tokens_per_category = None
        if "category" in df.columns:
            tokens_per_category = df.groupby("category")["num_tokens"].sum().to_dict()

        print(f"Total tokens in dataset: {total_tokens}")
        if tokens_per_category:
            print("Tokens per category:")
            for cat, tok in tokens_per_category.items():
                print(f"  {cat}: {tok}")
        else:
            print("No 'category' column found in dataset.")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)

        # Create DatasetDict with a single split
        dataset_dict = DatasetDict({"train": dataset})

        # Push to hub
        print(f"Uploading dataset to {repo_id}...")
        dataset_dict.push_to_hub(repo_id, token=self.token, private=private)

        # Create dataset card if requested
        if create_dataset_card:
            self._create_dataset_card(
                dataset_dir, repo_id, total_tokens, tokens_per_category
            )

        # Upload additional files (metadata, etc.)
        metadata_files = [
            dataset_dir / "dataset_metadata.json",
            dataset_dir / "file_metadata.csv",  # For OpenSubtitles
        ]

        for file_path in metadata_files:
            if file_path.exists():
                try:
                    self.api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=self.token,
                    )
                    print(f"Uploaded {file_path.name}")
                except Exception as e:
                    print(f"Error uploading {file_path.name}: {e}")

        print(
            f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}"
        )

    def _create_dataset_card(
        self,
        dataset_dir: Path,
        repo_id: str,
        total_tokens: int,
        tokens_per_category: Optional[dict],
    ) -> None:
        """Create a README.md dataset card."""
        import json

        # Load metadata
        metadata_path = dataset_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        config = metadata.get("config", {})

        # Determine size category based on number of documents
        num_documents = metadata.get("num_documents")
        if num_documents is None:
            # Try to infer from dataset files if not in metadata
            data_file = next(
                (
                    dataset_dir / f
                    for f in ["data.parquet", "data.csv"]
                    if (dataset_dir / f).exists()
                ),
                None,
            )
            if data_file is not None:
                import pandas as pd

                if str(data_file).endswith(".parquet"):
                    df = pd.read_parquet(data_file)
                else:
                    df = pd.read_csv(data_file)
                num_documents = len(df)
            else:
                num_documents = 0

        if num_documents < 1000:
            size_category = "n<1K"
        elif num_documents < 10000:
            size_category = "1K<=n<10K"
        elif num_documents < 100000:
            size_category = "10K<=n<100K"
        else:
            size_category = "n>=100K"

        # Helper to infer a field from config or documents
        def infer_field(field_name, hf_field=None):
            value = config.get(field_name)
            if value is not None:
                return value
            docs = metadata.get("documents")
            if docs and isinstance(docs, list):
                # Try to get all values for this field from documents
                doc_field = hf_field or field_name
                values = [
                    doc.get(doc_field)
                    for doc in docs
                    if doc_field in doc and doc.get(doc_field) is not None
                ]
                unique_values = sorted(set(values))
                if len(unique_values) == 0:
                    return "Unknown"
                if len(unique_values) > 3:
                    return "See individual files"
                return ", ".join(str(v) for v in unique_values)
            return "Unknown"

        language = config.get("language_code", "unknown")
        script = infer_field("script")
        category = infer_field("category")
        source = infer_field("data_source", "source")
        age_estimate = infer_field("age_estimate")

        # Create README content with correct YAML indentation
        readme_content = f"""---
task_categories:
- text-generation
language:
- {language}
license: {config.get('license', 'unknown')}
size_categories:
- {size_category}
dataset_info:
  features:
    - name: text
      dtype: string
    - name: category
      dtype: string
    - name: data-source
      dtype: string
    - name: script
      dtype: string
    - name: age-estimate
      dtype: string
    - name: license
      dtype: string
    - name: misc
      dtype: string
---

# {metadata.get('dataset_name', 'BabyLM Dataset')}

## Dataset Description

This dataset is part of the BabyLM multilingual collection.

### Dataset Summary

- **Language:** {language}
- **Script:** {script}
- **Category:** {category}
- **Source:** {source}
- **Age Estimate:** {age_estimate}
- **Number of Documents:** {num_documents}
- **Total Tokens:** {total_tokens}

### Tokens Per Category

"""

        if tokens_per_category:
            for cat, tok in tokens_per_category.items():
                readme_content += f"- **{cat}:** {tok} tokens\n"
        else:
            readme_content += "No category data available.\n"

        readme_content += f"""
### Data Fields

- `text`: The document text
- `category`: Type of content (e.g., child-directed-speech, educational, etc.)
- `data-source`: Original source of the data
- `script`: Writing system used
- `age-estimate`: Target age or age range
- `license`: Data license
- `misc`: Additional metadata (JSON string)

### Licensing Information

This dataset is licensed under: {config.get('license', 'See individual files')}

### Citation

Please cite the original data source: {config.get('data_source', 'Unknown')}
"""

        # Save README
        readme_path = dataset_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        # Upload README
        try:
            self.api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
            )
            print("Dataset card uploaded")
        except Exception as e:
            print(f"Error uploading README: {e}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Upload all BabyLM datasets in babylm_datasets directory."
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="multilingual-babylm/babylm_datasets",
        help="Path to the babylm_datasets directory.",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make uploaded repos private."
    )
    parser.add_argument(
        "--create_repo_if_missing",
        action="store_true",
        help="Create the repo if it does not exist.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use env if not provided).",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists() or not datasets_dir.is_dir():
        print(f"Datasets directory {datasets_dir} not found.")
        exit(1)

    uploader = HFDatasetUploader(token=args.token)

    for dataset_subdir in sorted(datasets_dir.iterdir()):
        if dataset_subdir.is_dir():
            repo_id = f"BabyLM-community/{dataset_subdir.name}"
            print(f"\nUploading {dataset_subdir} to {repo_id}...")
            try:
                uploader.upload_babylm_dataset(
                    dataset_dir=dataset_subdir,
                    repo_id=repo_id,
                    private=args.private,
                    create_dataset_card=True,
                    create_repo_if_missing=args.create_repo_if_missing,
                )
            except Exception as e:
                print(f"Failed to upload {dataset_subdir.name}: {e}")
