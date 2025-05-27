"""
HuggingFace dataset uploader for BabyLM datasets.
"""

import os
from pathlib import Path
from typing import Optional, Union
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
    
    def upload_babylm_dataset(self,
                             dataset_dir: Path,
                             repo_id: str,
                             private: bool = False,
                             create_dataset_card: bool = True,
                             create_repo_if_missing: bool = False) -> None:
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
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=private,
                    token=self.token,
                    exist_ok=True
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
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)
        
        # Create DatasetDict with a single split
        dataset_dict = DatasetDict({"train": dataset})
        
        # Push to hub
        print(f"Uploading dataset to {repo_id}...")
        dataset_dict.push_to_hub(
            repo_id,
            token=self.token,
            private=private
        )
        
        # Create dataset card if requested
        if create_dataset_card:
            self._create_dataset_card(dataset_dir, repo_id)
        
        # Upload additional files (metadata, etc.)
        metadata_files = [
            dataset_dir / "dataset_metadata.json",
            dataset_dir / "file_metadata.csv"  # For OpenSubtitles
        ]
        
        for file_path in metadata_files:
            if file_path.exists():
                try:
                    self.api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=self.token
                    )
                    print(f"Uploaded {file_path.name}")
                except Exception as e:
                    print(f"Error uploading {file_path.name}: {e}")
        
        print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    
    def _create_dataset_card(self, dataset_dir: Path, repo_id: str) -> None:
        """Create a README.md dataset card."""
        import json
        
        # Load metadata
        metadata_path = dataset_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        config = metadata.get("config", {})
        
        # Create README content
        readme_content = f"""---
            task_categories:
            - text-generation
            language:
            - {config.get('language_code', 'unknown')}
            license: {config.get('license', 'unknown')}
            size_categories:
            - n<1K
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

            - **Language:** {config.get('language_code', 'Unknown')}
            - **Script:** {config.get('script', 'Unknown')}
            - **Category:** {config.get('category', 'Unknown')}
            - **Source:** {config.get('data_source', 'Unknown')}
            - **Age Estimate:** {config.get('age_estimate', 'n/a')}
            - **Number of Documents:** {metadata.get('num_documents', 'Unknown')}

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
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Upload README
        try:
            self.api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token
            )
            print("Dataset card uploaded")
        except Exception as e:
            print(f"Error uploading README: {e}")