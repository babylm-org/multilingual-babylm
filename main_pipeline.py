"""
Main pipeline script for processing various data sources into BabyLM datasets.
Example usage for OpenSubtitles and other sources.
"""

from pathlib import Path
import argparse
import json

# Import our modules
from opensubtitles_processor import OpenSubtitlesProcessor
from babylm_dataset_builder import BabyLMDatasetBuilder, DatasetConfig
from hf_uploader import HFDatasetUploader


def process_opensubtitles(language_code: str, 
                         config_params: dict,
                         batch_size: int = 50,
                         upload: bool = False,
                         repo_id: str = None) -> None:
    """
    Process OpenSubtitles data for a specific language.
    
    Args:
        language_code: ISO 639-3 language code
        config_params: Dictionary with dataset configuration
        batch_size: Batch size for processing
        upload: Whether to upload to HuggingFace
        repo_id: HuggingFace repository ID
    """
    # Step 1: Process OpenSubtitles data
    print(f"Processing OpenSubtitles data for {language_code}...")
    processor = OpenSubtitlesProcessor(language_code)
    file_metadata_df, preprocessed_dir = processor.process_language(batch_size=batch_size)
    
    # Save complete file metadata (including XML metadata)
    file_metadata_path = processor.output_dir / f"{language_code}_complete_file_metadata.csv"
    file_metadata_df.to_csv(file_metadata_path, index=False)
    print(f"Complete file metadata saved to {file_metadata_path}")
    
    # Step 2: Build BabyLM dataset
    print("\nBuilding BabyLM dataset...")
    config = DatasetConfig(
        language_code=language_code,
        category="subtitles",  # OpenSubtitles is always subtitles
        data_source="OpenSubtitles",
        **config_params
    )
    
    builder = BabyLMDatasetBuilder(config)
    
    # Create metadata mapping from file metadata
    metadata_mapping = {}
    for _, row in file_metadata_df.iterrows():
        if row['processing_status'] == 'success':
            doc_metadata = {
                'year': row['year'],
                'folder_name': row['folder_name']
            }
            # Add any XML metadata fields
            for col in row.index:
                if col.startswith('meta_'):
                    doc_metadata[col] = row[col]
            metadata_mapping[row['file_id']] = doc_metadata
    
    # Add documents
    builder.add_documents_from_directory(preprocessed_dir, metadata_mapping)
    
    # Save dataset metadata
    builder.save_metadata()
    
    # Create final dataset table
    dataset_df = builder.create_dataset_table()
    print(f"\nDataset created with {len(dataset_df)} documents")
    
    # Step 3: Upload if requested
    if upload and repo_id:
        print(f"\nUploading to HuggingFace: {repo_id}")
        uploader = HFDatasetUploader()
        uploader.upload_babylm_dataset(
            dataset_dir=builder.output_dir,
            repo_id=repo_id
        )
    
    return builder.output_dir


def process_custom_source(texts_dir: Path,
                         language_code: str,
                         config_params: dict,
                         metadata_file: Path = None,
                         upload: bool = False,
                         repo_id: str = None) -> None:
    """
    Process custom text sources into BabyLM format.
    
    Args:
        texts_dir: Directory containing text files
        language_code: ISO 639-3 language code
        config_params: Dictionary with dataset configuration
        metadata_file: Optional JSON file with document metadata
        upload: Whether to upload to HuggingFace
        repo_id: HuggingFace repository ID
    """
    print(f"Processing custom source for {language_code}...")
    
    # Create config
    config = DatasetConfig(
        language_code=language_code,
        **config_params
    )
    
    # Build dataset
    builder = BabyLMDatasetBuilder(config)
    
    # Load metadata if provided
    metadata_mapping = {}
    if metadata_file and metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata_mapping = json.load(f)
    
    # Add documents
    builder.add_documents_from_directory(texts_dir, metadata_mapping)
    
    # Save and create dataset
    builder.save_metadata()
    dataset_df = builder.create_dataset_table()
    print(f"\nDataset created with {len(dataset_df)} documents")
    
    # Upload if requested
    if upload and repo_id:
        print(f"\nUploading to HuggingFace: {repo_id}")
        uploader = HFDatasetUploader()
        uploader.upload_babylm_dataset(
            dataset_dir=builder.output_dir,
            repo_id=repo_id
        )
    
    return builder.output_dir


def main():
    parser = argparse.ArgumentParser(description="Process data into BabyLM format")
    parser.add_argument("source", choices=["opensubtitles", "custom"],
                       help="Data source type")
    parser.add_argument("--language", "-l", required=True,
                       help="ISO 639-3 language code")
    
    # Common config parameters
    parser.add_argument("--script", required=True,
                       help="Script type (latin, cyrillic, arabic, etc.)")
    parser.add_argument("--age-estimate", required=True,
                       help="Age estimate (e.g., '4', '12-17', 'n/a')")
    parser.add_argument("--license", required=True,
                       help="License (e.g., cc-by, cc-by-sa)")
    parser.add_argument("--data-source", 
                       help="Data source name (for custom sources)")
    parser.add_argument("--category",
                       choices=["child-directed-speech", "educational", "child-books",
                               "child-wiki", "child-news", "subtitles", "qed",
                               "child-available-speech"],
                       help="Content category (for custom sources)")
    
    # Source-specific arguments
    parser.add_argument("--texts-dir", type=Path,
                       help="Directory with text files (for custom source)")
    parser.add_argument("--metadata-file", type=Path,
                       help="JSON file with document metadata (for custom source)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing (OpenSubtitles only)")
    
    # Upload arguments
    parser.add_argument("--upload", action="store_true",
                       help="Upload to HuggingFace after processing")
    parser.add_argument("--repo-id",
                       help="HuggingFace repo ID (e.g., 'username/babylm-eng')")
    
    # Optional parameters
    parser.add_argument("--source-url", help="Source URL")
    parser.add_argument("--source-identifier", help="Source identifier")
    parser.add_argument("--misc", type=json.loads,
                       help="Additional metadata as JSON string")
    
    args = parser.parse_args()
    
    # Prepare config parameters
    config_params = {
        "script": args.script,
        "age_estimate": args.age_estimate,
        "license": args.license,
    }
    
    if args.source_url:
        config_params["source_url"] = args.source_url
    if args.source_identifier:
        config_params["source_identifier"] = args.source_identifier
    if args.misc:
        config_params["misc"] = args.misc
    
    # Process based on source type
    if args.source == "opensubtitles":
        output_dir = process_opensubtitles(
            language_code=args.language,
            config_params=config_params,
            batch_size=args.batch_size,
            upload=args.upload,
            repo_id=args.repo_id
        )
    else:  # custom
        if not args.texts_dir or not args.texts_dir.exists():
            parser.error("--texts-dir is required for custom source")
        if not args.data_source:
            parser.error("--data-source is required for custom source")
        if not args.category:
            parser.error("--category is required for custom source")
        
        config_params["data_source"] = args.data_source
        config_params["category"] = args.category
        
        output_dir = process_custom_source(
            texts_dir=args.texts_dir,
            language_code=args.language,
            config_params=config_params,
            metadata_file=args.metadata_file,
            upload=args.upload,
            repo_id=args.repo_id
        )
    
    print(f"\nProcessing complete! Output directory: {output_dir}")


if __name__ == "__main__":
    main()