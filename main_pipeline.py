# main_pipeline.py
"""
Main pipeline script for processing various data sources into BabyLM datasets.
"""

from pathlib import Path
import argparse
import json
from typing import Optional, Dict, Any
from tqdm import tqdm

# Import our modules
from babylm_dataset_builder import BabyLMDatasetBuilder, DatasetConfig, DocumentConfig
from hf_uploader import HFDatasetUploader
from text_preprocessor import create_preprocessor, BasePreprocessor


def process_dataset(
    language_code: str,
    data_source: str,
    category: str,
    texts_dir: Path,
    document_config_params: dict,
    metadata_file: Optional[Path] = None,
    upload: bool = False,
    repo_id: Optional[str] = None,
    preprocessing_config: Optional[dict] = None,
    preprocessor_type: str = "text"
) -> Path:
    """
    Process any data source into BabyLM format.
    
    Args:
        language_code: ISO 639-3 language code
        data_source: Name of the data source
        category: Content category
        texts_dir: Directory containing text files
        document_config_params: Dictionary with document-level configuration
        metadata_file: Optional JSON file with document metadata
        upload: Whether to upload to HuggingFace
        repo_id: HuggingFace repository ID
        preprocessing_config: Configuration for text preprocessing
        preprocessor_type: Type of preprocessor to use
        
    Returns:
        Path to output directory
    """
    print(f"Processing {data_source} data for {language_code}...")
    
    # Create preprocessor if preprocessing is requested
    preprocessor = None
    if preprocessing_config:
        print(f"Using {preprocessor_type} preprocessor...")
        preprocessor = create_preprocessor(preprocessor_type, **preprocessing_config)
        
        # If preprocessor is provided, we need to preprocess the texts
        if preprocessor:
            preprocessed_dir = Path(f"./preprocessed_{data_source}_{language_code}")
            preprocessed_dir.mkdir(exist_ok=True)
            
            print("Preprocessing text files...")
            text_files = list(texts_dir.glob("*.txt"))
            
            for text_file in tqdm(text_files, desc="Preprocessing files"):
                output_file = preprocessed_dir / text_file.name
                try:
                    preprocessor.process_file(text_file, output_file)
                except Exception as e:
                    print(f"Error preprocessing {text_file}: {e}")
                    continue
            
            # Use preprocessed directory for dataset building
            texts_dir = preprocessed_dir
    
    # Create dataset config (just language code)
    dataset_config = DatasetConfig(language_code=language_code)
    
    # Create default document config
    default_doc_config = DocumentConfig(
        category=category,
        data_source=data_source,
        **document_config_params
    )
    
    # Build dataset
    builder = BabyLMDatasetBuilder(dataset_config)
    
    # Load metadata if provided
    metadata_mapping = {}
    if metadata_file and metadata_file.exists():
        print(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata_mapping = json.load(f)
    
    # Add documents
    builder.add_documents_from_directory(texts_dir, default_doc_config, metadata_mapping)
    
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
    parser = argparse.ArgumentParser(
        description="Process data into BabyLM format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--language", "-l", required=True,
                       help="ISO 639-3 language code")
    parser.add_argument("--data-source", "-s", required=True,
                       help="Data source name (e.g., OpenSubtitles, CHILDES, etc.)")
    parser.add_argument("--category", "-c", required=True,
                       choices=["child-directed-speech", "educational", "child-books",
                               "child-wiki", "child-news", "subtitles", "qed",
                               "child-available-speech"],
                       help="Content category")
    parser.add_argument("--texts-dir", "-t", type=Path, required=True,
                       help="Directory containing text files to process")
    
    # Dataset configuration
    parser.add_argument("--script", required=True,
                       help="Script type (latin, cyrillic, arabic, etc.)")
    parser.add_argument("--age-estimate", required=True,
                       help="Age estimate (e.g., '4', '12-17', 'n/a')")
    parser.add_argument("--license", required=True,
                       help="License (e.g., cc-by, cc-by-sa)")
    
    # Optional metadata
    parser.add_argument("--metadata-file", type=Path,
                       help="JSON file with document metadata")
    parser.add_argument("--source-url", help="Source URL")
    parser.add_argument("--source-identifier", help="Source identifier")
    parser.add_argument("--misc", type=json.loads,
                       help="Additional metadata as JSON string")
    
    # Upload arguments
    parser.add_argument("--upload", action="store_true",
                       help="Upload to HuggingFace after processing")
    parser.add_argument("--repo-id",
                       help="HuggingFace repo ID (e.g., 'username/babylm-eng')")
    
    # Preprocessing arguments
    parser.add_argument("--preprocess", action="store_true",
                       help="Enable text preprocessing")
    parser.add_argument("--preprocessor-type", default="text",
                       choices=["text", "subtitle", "transcript", "llm"],
                       help="Type of preprocessor to use")
    
    # Preprocessing options
    parser.add_argument("--lowercase", action="store_true", default=None,
                       help="Lowercase text")
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false",
                       help="Don't lowercase text")
    parser.add_argument("--fix-unicode", action="store_true", default=None,
                       help="Fix unicode issues with ftfy")
    parser.add_argument("--no-fix-unicode", dest="fix_unicode", action="store_false",
                       help="Don't fix unicode")
    parser.add_argument("--remove-timestamps", action="store_true", default=None,
                       help="Remove timestamps")
    parser.add_argument("--remove-stage-directions", action="store_true", default=None,
                       help="Remove [bracketed] stage directions")
    
    # LLM preprocessing options
    parser.add_argument("--llm-model", default="llama3.2",
                       help="Ollama model to use for LLM preprocessing")
    parser.add_argument("--llm-prompt",
                       help="Prompt for LLM filtering/processing")
    parser.add_argument("--llm-filter-threshold", type=float, default=0.7,
                       help="Threshold for LLM filtering (0-1)")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.texts_dir.exists():
        parser.error(f"Texts directory does not exist: {args.texts_dir}")
    
    if args.upload and not args.repo_id:
        parser.error("--repo-id is required when --upload is specified")
    
    # Prepare document config parameters
    document_config_params = {
        "script": args.script,
        "age_estimate": args.age_estimate,
        "license": args.license,
    }
    
    if args.source_url:
        document_config_params["source_url"] = args.source_url
    if args.source_identifier:
        document_config_params["source_identifier"] = args.source_identifier
    if args.misc:
        document_config_params["misc"] = args.misc
    
    # Prepare preprocessing config
    preprocessing_config = None
    if args.preprocess:
        preprocessing_config = {}
        
        # Add boolean options only if explicitly set
        if args.lowercase is not None:
            preprocessing_config["lowercase"] = args.lowercase
        if args.fix_unicode is not None:
            preprocessing_config["fix_unicode"] = args.fix_unicode
        if args.remove_timestamps is not None:
            preprocessing_config["remove_timestamps"] = args.remove_timestamps
        if args.remove_stage_directions is not None:
            preprocessing_config["remove_stage_directions"] = args.remove_stage_directions
        
        # Add LLM-specific options if using LLM preprocessor
        if args.preprocessor_type == "llm":
            if not args.llm_prompt:
                parser.error("--llm-prompt is required when using LLM preprocessor")
            preprocessing_config["model"] = args.llm_model
            preprocessing_config["prompt"] = args.llm_prompt
            preprocessing_config["filter_threshold"] = args.llm_filter_threshold
    
    # Process the dataset
    output_dir = process_dataset(
        language_code=args.language,
        data_source=args.data_source,
        category=args.category,
        texts_dir=args.texts_dir,
        document_config_params=document_config_params,
        metadata_file=args.metadata_file,
        upload=args.upload,
        repo_id=args.repo_id,
        preprocessing_config=preprocessing_config,
        preprocessor_type=args.preprocessor_type
    )
    
    print(f"\nProcessing complete! Output directory: {output_dir}")


if __name__ == "__main__":
    main()