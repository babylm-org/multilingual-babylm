"""
HuggingFace dataset uploader for BabyLM datasets.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, cast

import yaml
import hashlib
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from datasets.exceptions import DatasetNotFoundError
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer  # type: ignore
from pad_utils import get_byte_premium_factor, get_dataset_tier, get_dataset_size

TEMPLATE_PATH = Path("resources") / "readme_template.txt"
CONTRIBUTORS_PATH = Path("resources") / "contributors.yaml"
DATA_SOURCES_PATH = Path("resources") / "data_sources.yaml"


# Calculate token statistics
def count_tokens(text, tokenizer=None):
    if not isinstance(text, str):
        return 0
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    return len(text.split())


class HFDatasetUploader:
    """Handle uploading BabyLM datasets to HuggingFace."""

    def __init__(self, token: "Optional[str]" = None):
        load_dotenv()
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not found. Set it in .env or pass directly.")

        self.api = HfApi(token=self.token)

    def create_pr(self, repo_id, pr_title, pr_description):
        res = self.api.create_discussion(
            repo_id=repo_id,
            repo_type="dataset",
            title=pr_title,
            description=pr_description,
            pull_request=True,
        )
        print(f"Creating Pull Request with url {res.url}")
        pr_revision = f"refs/pr/{res.num}"
        return pr_revision

    def upload_babylm_dataset(
        self,
        language_code: str,
        script_code: str,
        dataset_dir: Path,
        repo_id: str,
        private: bool = True,
        create_dataset_card: bool = True,
        create_repo_if_missing: bool = True,
        add_to_existing_data: bool = False,
        tokenizer_name: Optional[str] = None,
        byte_premium_factor: Optional[float] = None,
        create_pr: Optional[bool] = False,
        pr_title: Optional[str] = "Update Dataset",
        pr_description: Optional[str] = "",
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

        # Fix schema issues before uploading
        if "misc" in df.columns:
            # Convert None/null values to json string
            df["misc"] = df["misc"].fillna("{}").astype(str)

        # assign language to documents
        df["language"] = language_code

        # most frequent script should be the given script
        # major_script = df["script"].mode()[0]
        # assert script_code == major_script

        # Assume hyphenated schema; compute num-tokens if missing
        if "num-tokens" not in df.columns:
            df["num-tokens"] = df["text"].apply(count_tokens, tokenizer=tokenizer)
        total_tokens = int(df["num-tokens"].sum())

        # get the dataset size in MB
        dataset_size = get_dataset_size(df)

        assert "category" in df.columns, "category must be defined"

        tokens_per_category = df.groupby("category")["num-tokens"].sum().to_dict()
        # scripts list
        scripts_list = sorted(
            {
                str(s).strip()
                for s in df.get("script", pd.Series()).astype(str).unique()
                if str(s).strip() and str(s).lower() != "nan"
            }
        )
        # group category counts
        tokens_per_group = self._compute_group_tokens(tokens_per_category)

        print(f"Total tokens in dataset: {total_tokens:,}")
        print("Tokens per category:")
        for cat, tok in tokens_per_category.items():
            print(f"  {cat}: {tok}")
        print("Tokens per group:")
        for g, tok in tokens_per_group.items():
            print(f"  {g}: {tok}")

        revision = None
        if create_pr:
            revision = self.create_pr(repo_id, pr_title, pr_description)

        # Create dataset card if requested
        if create_dataset_card:
            num_documents = len(df)
            self._create_dataset_card(
                dataset_dir=dataset_dir,
                repo_id=repo_id,
                total_tokens=total_tokens,
                tokens_per_category=tokens_per_category,
                num_documents=num_documents,
                dataset_size=dataset_size,
                major_script=script_code,
                scripts_list=scripts_list,
                language_code=language_code,
                tokens_per_group=tokens_per_group,
                tokenizer_name=tokenizer_name,
                byte_premium_factor=byte_premium_factor,
                revision=revision,
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
                        revision=revision,
                    )
                    print(f"Uploaded {file_path.name}")
                except Exception as e:
                    print(f"Error uploading {file_path.name}: {e}")

        # Define the expected features explicitly
        features = Features(
            {
                "text": Value("string"),
                "doc-id": Value("string"),
                "category": Value("string"),
                "data-source": Value("string"),
                "script": Value("string"),
                "age-estimate": Value("string"),
                "license": Value("string"),
                "misc": Value("string"),
                "num-tokens": Value("int64"),
                "language": Value("string"),
            }
        )

        dataset = Dataset.from_pandas(df, features=features)

        # Create DatasetDict with a single split
        dataset_dict = DatasetDict({"train": dataset})

        # Push to hub
        print(f"Uploading dataset to {repo_id}...")
        dataset_dict.push_to_hub(
            repo_id, token=self.token, private=private, revision=revision
        )

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
        dataset_size: float,
        major_script: str,
        language_code: str,
        scripts_list: Optional[List[str]] = None,
        tokens_per_group: Optional[Dict[str, int]] = None,
        tokenizer_name: Optional[str] = None,
        byte_premium_factor: Optional[float] = None,
        revision=None,
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
        num_documents = metadata.get("num_documents") or num_documents  # type: ignore
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
        # Script list handling
        if not scripts_list or len(scripts_list) == 0:
            script_display = "Unknown"
        else:
            script_display = ", ".join(scripts_list)

        if byte_premium_factor is None:
            byte_premium_factor = get_byte_premium_factor(language, major_script)

        # calculate dataset_tier, allowing for at most 1% difference from the required size
        dataset_tier, expected_size, _ = get_dataset_tier(
            dataset_size, byte_premium_factor, percent_tolerance=0.01
        )
        license_metadata = config.get("license", "unknown")
        data_source = config.get("data_source", "Unknown")

        tokens_per_category_content = ""
        if tokens_per_category:
            for cat, tok in tokens_per_category.items():
                tokens_per_category_content += f"- **{cat}:** {tok:,} tokens\n"
        else:
            tokens_per_category_content += "No category data available.\n"

        # Tokens per group section
        tokens_per_category_content += "\n### Tokens Per Group\n\n"
        if tokens_per_group:
            for grp, tok in tokens_per_group.items():
                tokens_per_category_content += f"- **{grp}:** {tok:,} tokens\n"
        else:
            tokens_per_category_content += "No group data available.\n"

        with TEMPLATE_PATH.open("r") as f:
            readme_content = f.read()

        dataset_name = metadata.get("dataset_name", "BabyLM Dataset")

        if tokenizer_name is None:
            tokenizer_name = "separate by whitespace"

        # create contributors section
        with CONTRIBUTORS_PATH.open("r") as f:
            contributors = yaml.safe_load(f)

        contributors_lang = contributors.get(language_code)
        if contributors_lang is None:
            contributors_readme = "n/a"
        else:
            contributors_readme = ""
            contributors_lang = sorted(contributors_lang, key=lambda x: x["name"])
            for contributor in contributors_lang:
                if contributor.get("mail") is not None:
                    contributors_readme += (
                        f"* {contributor['name']} ({contributor['mail']})\n"
                    )
                else:
                    contributors_readme += f"* {contributor['name']}\n"

        # create data sources section
        with DATA_SOURCES_PATH.open("r") as f:
            sources = yaml.safe_load(f)
        sources_lang = sources.get(language_code)
        if sources_lang is None:
            sources_lang = "n/a"
        else:
            sources_df = pd.DataFrame(sources_lang)
            group_readmes = []
            for category, group in sources_df.groupby("category"):
                group_readme = f"#### {category}\n"
                group_items = []
                for _, source in group.iterrows():
                    item = f"- {source['name']}"
                    if "link" in source:
                        item += f"\n\t - source: {source['link']}"
                    if "description" in source:
                        item += f"\n\t - description: {source['description']}"
                    if "citation" in source:
                        item += f"\n\t - citation: {source['citation']}"
                    group_items.append(item)
                group_readme += "\n".join(group_items)
                group_readmes.append(group_readme)

            data_sources_readme = "\n\n".join(group_readmes)

        # format readme
        readme_content = readme_content.format(
            language=language,
            license_metadata=license_metadata,
            size_category=size_category,
            script_display=script_display,
            dataset_tier=dataset_tier,
            dataset_size=dataset_size,
            expected_size=expected_size,
            byte_premium_factor=byte_premium_factor,
            num_documents=num_documents,
            total_tokens=total_tokens,
            tokens_per_category_content=tokens_per_category_content,
            data_source=data_source,
            dataset_name=dataset_name,
            tokenizer_name=tokenizer_name,
            contributors_readme=contributors_readme,
            data_sources_readme=data_sources_readme,
        )

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
                revision=revision,
            )
            print("Dataset card uploaded")
        except Exception as e:
            print(f"Error uploading README: {e}")

    def update_all_readmes(
        self,
        repo_ids: Optional[list[str]] = None,
        check_empty: bool = True,
        byte_premium_factor: Optional[float] = None,
    ):
        """Bulk update README files for all BabyLM language datasets discovered dynamically.

        Discovery logic:
          1. List all datasets for author 'BabyLM-community'. (Token required to include private repos.)
          2. Filter IDs starting with 'BabyLM-community/babylm-'.
          3. Iterate each repo and regenerate README with scripts list + grouped category counts.
        """
        if repo_ids is None:
            repo_ids = self._discover_babylm_repos(check_empty=check_empty)

        tokenizers = {
            "jpn": "tohoku-nlp/bert-base-japanese",
            "zho": "Qwen/Qwen3-0.6B",
            "yue": "Qwen/Qwen1.5-7B-Chat",
            "kor": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        }

        if not repo_ids:
            print("No BabyLM datasets found with prefix 'BabyLM-community/babylm-'.")
            return

        print(f"Discovered {len(repo_ids)} BabyLM dataset repos to update.")
        for repo_id in repo_ids:
            language_code = repo_id.split("-")[-1]
            tokenizer_name = tokenizers.get(language_code, None)

            suffix = repo_id.split("babylm-")[-1]
            print(f"Updating README for {repo_id}...")
            try:
                ds = load_dataset(repo_id, split="train", token=self.token)
                df_obj = ds.to_pandas()  # type: ignore[attr-defined]
                assert isinstance(df_obj, pd.DataFrame), (
                    "Expected pandas DataFrame from dataset"
                )
                df: pd.DataFrame = cast(pd.DataFrame, df_obj)
            except Exception as e:
                print(f"  Could not load dataset: {e}")
                continue
            if "num-tokens" not in df.columns:
                tokenizer = (
                    AutoTokenizer.from_pretrained(tokenizer_name)
                    if tokenizer_name
                    else None
                )
                df["num-tokens"] = df["text"].apply(count_tokens, tokenizer=tokenizer)
            total_tokens = int(df["num-tokens"].sum())
            if "category" not in df.columns:
                print("  Missing 'category' column; skipping.")
                continue
            tokens_per_category = df.groupby("category")["num-tokens"].sum().to_dict()
            if "script" in df.columns:
                scripts_list = sorted(
                    {
                        str(s).strip()
                        for s in df["script"].astype(str).unique()
                        if str(s).strip() and str(s).lower() != "nan"
                    }
                )
            else:
                scripts_list = []

            # median script value is the dominating script
            major_script = df["script"].mode()[0]

            tokens_per_group = self._compute_group_tokens(tokens_per_category)
            dataset_size = get_dataset_size(df)
            tmp_dir = Path(f"_tmp_readme_{suffix}")
            tmp_dir.mkdir(exist_ok=True)

            self._create_dataset_card(
                dataset_dir=tmp_dir,
                repo_id=repo_id,
                total_tokens=total_tokens,
                tokens_per_category=tokens_per_category,
                num_documents=len(df),
                dataset_size=dataset_size,
                scripts_list=scripts_list,
                major_script=major_script,
                language_code=language_code,
                tokens_per_group=tokens_per_group,
                tokenizer_name=tokenizer_name,
                byte_premium_factor=byte_premium_factor,
            )
            try:
                (tmp_dir / "README.md").unlink()
                tmp_dir.rmdir()
            except Exception:
                pass

    # Migration helpers removed: migration is complete and schema is canonical.

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
            ds_id = getattr(d, "id", None)
            if (
                not isinstance(ds_id, str)
                or not ds_id.startswith(prefix)
                or "subtitles" in ds_id
            ):
                continue
            # Check archive/deprecation indicators
            tags = set(getattr(d, "tags", []) or [])
            card_data = getattr(d, "cardData", {}) or {}
            archived_flag = False
            if isinstance(card_data, dict) and card_data.get("archived") is True:
                archived_flag = True
            if "archived" in tags or "deprecated" in tags:
                archived_flag = True
            if archived_flag:
                print(f"Skipping archived/deprecated dataset: {ds_id}")
                continue
            candidates.append(ds_id)
        active: List[str] = []
        for repo_id in sorted(set(candidates)):
            if check_empty:
                try:
                    ds = load_dataset(repo_id, split="train", token=self.token)
                    # Quickly assess emptiness (cast for type checker)
                    from datasets import (
                        Dataset as HFDataset,
                    )  # local import to avoid top-level clash

                    if len(cast(HFDataset, ds)) == 0:  # type: ignore[arg-type]
                        print(f"Skipping empty dataset: {repo_id}")
                        continue
                except Exception as e:
                    print(f"Skipping dataset (load failed): {repo_id} ({e})")
                    continue
            print(f"Discovered repo: {repo_id}")
            active.append(repo_id)

        return active


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage BabyLM language datasets on the Hub (update READMEs)."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use env if not provided).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Update a specific BabyLM repo, specified with --repo_id",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Don't check if repo is empty (reduce time in repo discovery)",
    )
    parser.add_argument(
        "--byte-premium-factor",
        type=float,
        default=None,
        help="Provide byte-premium factor manually, instead of retrieving it automatically (override).",
    )

    args = parser.parse_args()
    uploader = HFDatasetUploader(token=args.token)
    if args.repo_id is None:
        uploader.update_all_readmes(
            check_empty=not args.no_check, byte_premium_factor=args.byte_premium_factor
        )
    else:
        uploader.update_all_readmes(
            repo_ids=[args.repo_id],
            check_empty=not args.no_check,
            byte_premium_factor=args.byte_premium_factor,
        )
