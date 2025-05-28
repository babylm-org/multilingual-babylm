"""
General dataset builder for BabyLM multilingual datasets.
This can be used for any data source, not just OpenSubtitles.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DatasetConfig:
    """Configuration for a BabyLM dataset."""
    language_code: str  # ISO 639-3 code
    category: str  # One of the predefined categories
    data_source: str
    script: str  # latin, cyrillic, etc.
    age_estimate: str  # Specific age, range, or "n/a"
    license: str  # cc-by, cc-by-sa, etc.
    misc: Optional[Dict[str, str]] = None
    source_url: Optional[str] = None
    source_identifier: Optional[str] = None
    
    @property
    def dataset_name(self) -> str:
        return f"babylm-{self.language_code}"
    
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
            "child-available-speech"
        }
        if self.category not in allowed_categories:
            raise ValueError(f"Category must be one of: {allowed_categories}")


class BabyLMDatasetBuilder:
    """Build and manage BabyLM datasets from various sources."""
    
    def __init__(self, config: DatasetConfig, output_dir: Path = Path("./babylm_datasets")):
        self.config = config
        self.config.validate_category()
        
        self.output_dir = output_dir / self.config.dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.texts_dir = self.output_dir / "texts"
        self.texts_dir.mkdir(exist_ok=True)
        
        self.metadata_path = self.output_dir / "dataset_metadata.json"
        self.documents: List[Dict] = []
        
    def add_document(self, 
                    text: str, 
                    document_id: str,
                    additional_metadata: Optional[Dict] = None) -> None:
        """Add a document to the dataset."""
        # Save text file
        text_path = self.texts_dir / f"{document_id}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Create document metadata
        doc_metadata = {
            "document_id": document_id,
            "text_file": str(text_path.relative_to(self.output_dir)),
            "category": self.config.category,
            "data_source": self.config.data_source,
            "script": self.config.script,
            "age_estimate": self.config.age_estimate,
            "license": self.config.license,
        }
        
        # Add optional fields
        if self.config.source_url:
            doc_metadata["source_url"] = self.config.source_url
        if self.config.source_identifier:
            doc_metadata["source_identifier"] = self.config.source_identifier
        if self.config.misc:
            doc_metadata["misc"] = self.config.misc
        if additional_metadata:
            doc_metadata["additional_metadata"] = additional_metadata
            
        self.documents.append(doc_metadata)
    
    def add_documents_from_directory(self, 
                                   texts_dir: Path,
                                   metadata_mapping: Optional[Dict[str, Dict]] = None) -> None:
        """
        Add multiple documents from a directory of text files.
        
        Args:
            texts_dir: Directory containing text files
            metadata_mapping: Optional dict mapping document_id to additional metadata
        """
        metadata_mapping = metadata_mapping or {}
        
        for text_file in texts_dir.glob("*.txt"):
            document_id = text_file.stem
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            additional_metadata = metadata_mapping.get(document_id, {})
            self.add_document(text, document_id, additional_metadata)
    
    def save_metadata(self) -> None:
        """Save dataset metadata to JSON file."""
        metadata = {
            "dataset_name": self.config.dataset_name,
            "language_code": self.config.language_code,
            "creation_date": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "config": asdict(self.config),
            "documents": self.documents
        }
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Metadata saved to {self.metadata_path}")
    
    def create_dataset_table(self) -> pd.DataFrame:
        """Create the standardized BabyLM dataset table."""
        rows = []
        
        for doc in self.documents:
            # Read the text
            text_path = self.output_dir / doc["text_file"]
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            row = {
                "text": text,
                "category": doc["category"],
                "data-source": doc["data_source"],
                "script": doc["script"],
                "age-estimate": doc["age_estimate"],
                "license": doc["license"]
            }
            
            # Add misc field if present
            if "misc" in doc and doc["misc"]:
                row["misc"] = json.dumps(doc["misc"])
            else:
                row["misc"] = None
                
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save as CSV and parquet for flexibility
        csv_path = self.output_dir / f"{self.config.dataset_name}_dataset.csv"
        parquet_path = self.output_dir / f"{self.config.dataset_name}_dataset.parquet"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        
        print(f"Dataset table saved to:")
        print(f"  - {csv_path}")
        print(f"  - {parquet_path}")
        
        return df
    
    def get_upload_ready_dataset(self) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Get the dataset in a format ready for upload to HuggingFace.
        
        Returns:
            Dict with 'data' (DataFrame) and 'metadata' (Dict)
        """
        df = self.create_dataset_table()
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            "data": df,
            "metadata": metadata
        }