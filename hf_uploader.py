"""
HuggingFace dataset uploader for BabyLM datasets.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, cast

import hashlib
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer # type: ignore


class HFDatasetUploader:
    """Handle uploading BabyLM datasets to HuggingFace."""

    def __init__(self, token: "Optional[str]" = None):
        load_dotenv()
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not found. Set it in .env or pass directly.")

        self.api = HfApi(token=self.token)

    def upload_babylm_dataset(
        self,
        language_code: str,
        dataset_dir: Path,
        repo_id: str,
        private: bool = True,
        create_dataset_card: bool = True,
        create_repo_if_missing: bool = True,
        add_to_existing_data: bool = False,
        tokenizer_name: Optional[str] = None,
    ) -> None:
        """
        Upload a BabyLM dataset to HuggingFace.

        Args:
            dataset_dir: Directory containing the dataset files
            repo_id: HuggingFace repo ID (e.g., "username/babylm-eng")
            private: Whether to make the repo private
            create_dataset_card: Whether to create a README
            create_repo_if_missing: Whether to create the repo if it doesn't exist
            add_to_existing_data: Add to the existing dataset if it already exists. Overrides previous data if set to False.

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
            csv_dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(csv_dfs, ignore_index=True)
        else:
            raise ValueError(f"No dataset files found in {dataset_dir}")

        if add_to_existing_data:
            try:
                prev_ds = load_dataset(repo_id, token=self.token, split="train")
                # to_pandas is available at runtime; ignore static checker complaints
                prev_data = prev_ds.to_pandas()  # type: ignore[attr-defined]
                if isinstance(prev_data, pd.DataFrame) and isinstance(df, pd.DataFrame):
                    print(
                        f"Merging with existing data from {repo_id} (rows: {len(prev_data)})..."
                    )
                    df = pd.concat([prev_data, df], ignore_index=True)
                    print("Running deduplication on merged dataset...")
                    df = self._deduplicate_by_text(df)
                else:
                    raise TypeError(
                        "Both previous data and new data must be pandas DataFrames."
                    )
            except DatasetNotFoundError:
                print("Previous data not found")

        tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None
        )

        # Calculate token statistics
        def count_tokens(text, tokenizer=None):
            if not isinstance(text, str):
                return 0
            if tokenizer:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            return len(text.split())

        # Fix schema issues before uploading
        if "misc" in df.columns:
            # Convert None/null values to json string
            df["misc"] = df["misc"].fillna("{}").astype(str)


        # assign language to documents
        df["language"] = language_code
        df["num_tokens"] = df["text"].apply(count_tokens, tokenizer=tokenizer)
        total_tokens = int(df["num_tokens"].sum())

        assert "category" in df.columns, "category must be defined"


        tokens_per_category = df.groupby("category")["num_tokens"].sum().to_dict()
        # NEW: scripts list
        scripts_list = sorted(
            {
                str(s).strip()
                for s in df.get("script", pd.Series()).astype(str).unique()
                if str(s).strip() and str(s).lower() != "nan"
            }
        )
        # NEW: group category counts
        tokens_per_group = self._compute_group_tokens(tokens_per_category)

        print(f"Total tokens in dataset: {total_tokens}")
        print("Tokens per category:")
        for cat, tok in tokens_per_category.items():
            print(f"  {cat}: {tok}")
        print("Tokens per group:")
        for g, tok in tokens_per_group.items():
            print(f"  {g}: {tok}")

        # Define the expected features explicitly
        features = Features(
            {
                "text": Value("string"),
                "doc_id": Value("string"),
                "category": Value("string"),
                "data-source": Value("string"),
                "script": Value("string"),
                "age-estimate": Value("string"),
                "license": Value("string"),
                "misc": Value("string"),
                "num_tokens": Value("int64"),
                "language": Value("string"),
            }
        )

        dataset = Dataset.from_pandas(df, features=features)

        # Create DatasetDict with a single split
        dataset_dict = DatasetDict({"train": dataset})

        # Push to hub
        print(f"Uploading dataset to {repo_id}...")
        dataset_dict.push_to_hub(repo_id, token=self.token, private=private)

        # Create dataset card if requested
        if create_dataset_card:
            num_documents = len(df)
            self._create_dataset_card(
                dataset_dir=dataset_dir,
                repo_id=repo_id,
                total_tokens=total_tokens,
                tokens_per_category=tokens_per_category,
                num_documents=num_documents,
                scripts_list=scripts_list,
                tokens_per_group=tokens_per_group,
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

    def _compute_group_tokens(
        self, tokens_per_category: Dict[str, int]
    ) -> Dict[str, int]:
        """Compute tokens per high-level group based on predefined mapping."""
        padding_cats = [
            c
            for c in tokens_per_category
            if c.startswith("padding") or c == "simplified-text"
        ]
        category_map = {
            "Transcription": ["child-directed-speech", "child-available-speech"],
            "Education": ["educational"],
            "Books, Wiki, News": ["child-books", "child-wiki", "child-news"],
            "Subtitles": ["subtitles", "qed"],
            "Padding": padding_cats,
        }
        group_tokens = {}
        for group, cats in category_map.items():
            group_tokens[group] = int(sum(tokens_per_category.get(c, 0) for c in cats))
        return group_tokens

    def _deduplicate_by_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate documents based on the hash of the text field.
        Keeps the first occurrence of each unique text.
        """
        before = len(df)
        df["text_hash"] = df["text"].apply(
            lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
        )
        df = df.drop_duplicates(subset=["text_hash"]).drop(columns=["text_hash"])
        after = len(df)
        print(
            f"Deduplicated merged dataset: removed {before - after} duplicates, {after} remain."
        )
        return df

    def _create_dataset_card(
        self,
        dataset_dir: Path,
        repo_id: str,
        total_tokens: int,
        tokens_per_category: Optional[dict],
        num_documents: int,
        scripts_list: Optional[List[str]] = None,
        tokens_per_group: Optional[Dict[str, int]] = None,
    ) -> None:
        """Create (or overwrite) a README.md dataset card and upload it."""
        import json

        # Load metadata
        metadata_path = dataset_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            language_code = repo_id.split("-")[-1]
            metadata = {"config": {"language_code": language_code}}

        config = metadata.get("config", {})

        # Determine size category based on number of documents
        num_documents = metadata.get("num_documents") or num_documents # type: ignore
        md_num_docs = metadata.get("num_documents")
        if isinstance(md_num_docs, int):
            num_documents = md_num_docs
        elif isinstance(md_num_docs, str) and md_num_docs.isdigit():
            try:
                num_documents = int(md_num_docs)
            except ValueError:
                pass
        # size category
        if num_documents < 1_000:
            size_category = "n<1K"
        elif num_documents < 10_000:
            size_category = "1K<n<10K"
        elif num_documents < 100_000:
            size_category = "10K<n<100K"
        elif num_documents < 1_000_000:
            size_category = "100K<n<1M"
        elif num_documents < 10_000_000:
            size_category = "1M<n<10M"
        elif num_documents < 100_000_000:
            size_category = "10M<n<100M"
        else:
            size_category = "100M<n<1B"

        language = config.get("language_code", "unknown")
        # NEW script list handling
        if not scripts_list or len(scripts_list) == 0:
            script_display = "Unknown"
        else:
            script_display = ", ".join(scripts_list)

        # Create README content with correct YAML indentation
        readme_content = f"""---
task_categories:
- text-generation
language:
- {language}
license: {config.get("license", "unknown")}
size_categories:
- {size_category}
dataset_info:
  features:
    - name: text
      dtype: string
    - name: doc_id
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
    - name: num_tokens
      dtype: int64
    - name: language
      dtype: string
---

# {metadata.get("dataset_name", "BabyLM Dataset")}

## Dataset Description

This dataset is part of the BabyLM multilingual collection.

### Dataset Summary

- **Language:** {language}
- **Script:** {script_display}
- **Number of Documents:** {num_documents}
- **Total Tokens:** {total_tokens}

### Tokens Per Category

"""

        if tokens_per_category:
            for cat, tok in tokens_per_category.items():
                readme_content += f"- **{cat}:** {tok} tokens\n"
        else:
            readme_content += "No category data available.\n"

        # NEW: Tokens per group section
        readme_content += "\n### Tokens Per Group\n\n"
        if tokens_per_group:
            for grp, tok in tokens_per_group.items():
                readme_content += f"- **{grp}:** {tok} tokens\n"
        else:
            readme_content += "No group data available.\n"

        readme_content += f"""
### Data Fields

- `text`: The document text
- `doc_id`: Unique identifier for the document
- `category`: Type of content (e.g., child-directed-speech, educational, etc.)
- `data-source`: Original source of the data
- `script`: Writing system used (ISO 15924)
- `age-estimate`: Target age or age range
- `license`: Data license
- `misc`: Additional metadata (JSON string)
- `num_tokens`: Number of tokens per item (based on white-space split)
- `language`: Language code (ISO 639-3)

### Licensing Information

This dataset is licensed under: {config.get("license", "See individual files")}

### Citation

Please cite the original data source: {config.get("data_source", "Unknown")}
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

    def update_all_readmes(self):
        """Bulk update README files for all BabyLM language datasets discovered dynamically.

        Discovery logic:
          1. List all datasets for author 'BabyLM-community'. (Token required to include private repos.)
          2. Filter IDs starting with 'BabyLM-community/babylm-'.
          3. Iterate each repo and regenerate README with scripts list + grouped category counts.
        """
        repo_ids = self._discover_babylm_repos()
        if not repo_ids:
            print("No BabyLM datasets found with prefix 'BabyLM-community/babylm-'.")
            return
        print(f"Discovered {len(repo_ids)} BabyLM dataset repos to update.")
        for repo_id in repo_ids:
            suffix = repo_id.split("babylm-")[-1]
            print(f"Updating README for {repo_id}...")
            try:
                ds = load_dataset(repo_id, split="train", token=self.token)
                df_obj = ds.to_pandas()  # type: ignore[attr-defined]
                assert isinstance(df_obj, pd.DataFrame), "Expected pandas DataFrame from dataset"
                df: pd.DataFrame = cast(pd.DataFrame, df_obj)
            except Exception as e:
                print(f"  Could not load dataset: {e}")
                continue
            if "num_tokens" not in df.columns:
                df["num_tokens"] = df["text"].apply(lambda x: len(str(x).split()))
            total_tokens = int(df["num_tokens"].sum())
            if "category" not in df.columns:
                print("  Missing 'category' column; skipping.")
                continue
            tokens_per_category = df.groupby("category")["num_tokens"].sum().to_dict()
            if "script" in df.columns:
                scripts_list = sorted({
                    str(s).strip() for s in df["script"].astype(str).unique()
                    if str(s).strip() and str(s).lower() != "nan"
                })
            else:
                scripts_list = []
            tokens_per_group = self._compute_group_tokens(tokens_per_category)
            tmp_dir = Path(f"_tmp_readme_{suffix}")
            tmp_dir.mkdir(exist_ok=True)
            self._create_dataset_card(
                dataset_dir=tmp_dir,
                repo_id=repo_id,
                total_tokens=total_tokens,
                tokens_per_category=tokens_per_category,
                num_documents=len(df),
                scripts_list=scripts_list,
                tokens_per_group=tokens_per_group,
            )
            try:
                (tmp_dir / "README.md").unlink()
                tmp_dir.rmdir()
            except Exception:
                pass

    def _discover_babylm_repos(self, check_empty: bool = True) -> List[str]:
        """Return sorted list of active (non-archived, non-empty) BabyLM dataset repo_ids.

        Filtering steps:
          1. List all datasets for author 'BabyLM-community' (token ensures private visibility).
          2. Keep ids starting with 'BabyLM-community/babylm-'.
          3. Exclude repos marked archived/deprecated (by tag or cardData flag).
          4. Exclude repos whose train split cannot be loaded or has zero rows.
        """
        prefix = "BabyLM-community/babylm-"
        try:
            all_ds = self.api.list_datasets(author="BabyLM-community")
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return []
        candidates: List[str] = []
        for d in all_ds:
            ds_id = getattr(d, 'id', None)
            if not isinstance(ds_id, str) or not ds_id.startswith(prefix):
                continue
            # Check archive/deprecation indicators
            tags = set(getattr(d, 'tags', []) or [])
            card_data = getattr(d, 'cardData', {}) or {}
            archived_flag = False
            if isinstance(card_data, dict) and card_data.get('archived') is True:
                archived_flag = True
            if 'archived' in tags or 'deprecated' in tags:
                archived_flag = True
            if archived_flag:
                print(f"Skipping archived/deprecated dataset: {ds_id}")
                continue
            candidates.append(ds_id)
        active: List[str] = []
        for repo_id in sorted(set(candidates)):
            if check_empty:
                try:
                    ds = load_dataset(repo_id, split='train', token=self.token)
                    # Quickly assess emptiness (cast for type checker)
                    from datasets import Dataset as HFDataset  # local import to avoid top-level clash
                    if len(cast(HFDataset, ds)) == 0:  # type: ignore[arg-type]
                        print(f"Skipping empty dataset: {repo_id}")
                        continue
                except Exception as e:
                    print(f"Skipping dataset (load failed): {repo_id} ({e})")
                    continue
            active.append(repo_id)

        return active


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bulk update BabyLM language dataset READMEs (scripts list + grouped category counts)."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use env if not provided).",
    )
    args = parser.parse_args()
    uploader = HFDatasetUploader(token=args.token)
    uploader.update_all_readmes()
