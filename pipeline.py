# pipeline.py
"""
Main pipeline script for processing various data sources into BabyLM datasets.
"""

from pathlib import Path
import argparse
import json
from typing import Optional

# Import our modules
from babylm_dataset_builder import BabyLMDatasetBuilder, DatasetConfig
from hf_uploader import HFDatasetUploader
from preprocessor import preprocess_dataset
from language_filter import LanguageFilter, print_filtering_results
from language_scripts import validate_script_code
from loader import get_loader


def process_dataset(
    language_code: str,
    data_path: Path,
    document_config_params: dict,
    metadata_file: Optional[Path] = None,
    upload: bool = False,
    repo_id: Optional[str] = None,
    preprocess_text: bool = False,
    data_type: str = "text",
    enable_language_filtering: bool = False,
    language_filter_threshold: float = 0.8,
    tokenizer_name: Optional[str] = None,
) -> Path:
    """
    Process any data source into BabyLM format.

    Args:
        language_code: ISO 639-3 language code
        data_path: Path to data directory or file
        document_config_params: Dictionary with document-level configuration
        metadata_file: Optional JSON file with document metadata
        upload: Whether to upload to HuggingFace
        repo_id: HuggingFace repository ID
        preprocess_text: Enable text preprocessing
        data_type: Type of loader to use
        enable_language_filtering: Whether to enable language filtering
        language_filter_threshold: Minimum confidence for language filtering
        tokenizer_name: Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)

    Returns:
        Path to output directory
    """
    print(f"Processing data for {language_code}...")

    # 1. Load data using loader
    loader = get_loader(data_type)
    docs = loader.load_data(data_path)

    # 2. Load metadata file if provided and merge
    metadata_mapping = {}
    if metadata_file and metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_mapping = json.load(f)
    # Metadata file overrides document-level metadata
    for doc in docs:
        doc_id = doc["doc_id"]
        if doc_id in metadata_mapping:
            doc["metadata"].update(metadata_mapping[doc_id])

    # 3. Build dataset
    dataset_config = DatasetConfig(language_code=language_code)
    # Zzzz : default ISO 15924 value for Unknown or Unencoded
    script_code = document_config_params.get("script", "Zzzz") 
    if not validate_script_code(script_code):
        raise ValueError(
            f"Invalid script code '{script_code}'. Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)"
        )
    builder = BabyLMDatasetBuilder(dataset_config)
    builder.add_documents_from_iterable(docs, document_config_params)
    dataset_df = builder.create_dataset_table()

    # 4. Preprocess all texts (if requested)
    if preprocess_text:
        dataset_df = preprocess_dataset(dataset_df)
        builder.dataset_table = dataset_df

    # 5. Language filtering if enabled
    if enable_language_filtering:
        lang_filter = LanguageFilter()
        filter_results = lang_filter.filter_documents(
            builder.dataset_table,
            expected_language=language_code,
            expected_script=script_code,
            min_confidence=language_filter_threshold,
        )
        print_filtering_results(filter_results, language_code, script_code)
        # Only keep matching documents
        matching_ids = set(filter_results["match_ids"])
        builder.dataset_table = builder.dataset_table[
            builder.dataset_table["doc_id"].isin(matching_ids)
        ]

    # 6. Save and create dataset
    builder.save_dataset()
    print(f"\nDataset created with {len(builder.dataset_table)} documents")

    # 7. Upload if requested
    if upload and repo_id:
        print(f"\nUploading to HuggingFace: {repo_id}")
        uploader = HFDatasetUploader()
        uploader.upload_babylm_dataset(
            dataset_dir=builder.output_dir,
            repo_id=repo_id,
            create_repo_if_missing=True,
            tokenizer_name=tokenizer_name,
        )

    return builder.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Process data into BabyLM format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required arguments
    parser.add_argument(
        "--language", "-l", required=True, help="ISO 639-3 language code"
    )
    parser.add_argument(
        "--script", required=True, help="Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)"
    )
    parser.add_argument(
        "--data-path",
        "-t",
        type=Path,
        required=True,
        help="Path to data directory or file",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        choices=["text", "json", "jsonl", "csv", "hf"],
        help="Loader type for input data",
    )

    # Optional arguments, supplement missing values in documents
    parser.add_argument(
        "--data-source",
        "-s",
        type=str,
        help="Data source name (e.g., OpenSubtitles, CHILDES, etc.)",
    )
    parser.add_argument(
        "--category",
        "-c",
        choices=[
            "child-directed-speech",
            "educational",
            "child-books",
            "child-wiki",
            "child-news",
            "subtitles",
            "qed",
            "child-available-speech",
            "simplified-text",
        ],
        help="Content category",
    )
    parser.add_argument(
        "--age-estimate",
        type=str,
        help="Age estimate (e.g., '4', '12-17', 'n/a')",
    )
    parser.add_argument(
        "--license", help="License (e.g., cc-by, cc-by-sa)"
    )
    parser.add_argument(
        "--misc", type=json.loads, help="Additional metadata as JSON string"
    )
    
    parser.add_argument(
        "--metadata-file", type=Path, help="JSON file with document metadata"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace after processing"
    )
    parser.add_argument(
        "--repo-id", help="HuggingFace repo ID (e.g., 'username/babylm-eng')"
    )
    parser.add_argument(
        "--enable-language-filtering",
        action="store_true",
        help="Enable language and script filtering using GlotLID v3",
    )
    parser.add_argument(
        "--language-filter-threshold",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for language filtering (0.0-1.0)",
    )
    parser.add_argument(
        "--preprocess-text", action="store_true", help="Enable text preprocessing"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)",
    )

    args = parser.parse_args()

    if not validate_script_code(args.script):
        raise ValueError(
            f"Invalid script code '{args.script}'. Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)"
        )


    document_config_params = {
        "script": args.script,
    }
    if args.data_source:
        document_config_params["data_source"] = args.data_source
    if args.category:
        document_config_params["category"] = args.category
    if args.age_estimate:
        document_config_params["age_estimate"] = args.age_estimate
    if args.license:
        document_config_params["license"] = args.license
    if args.misc:
        document_config_params["misc"] = args.misc

    process_dataset(
        language_code=args.language,
        data_path=args.data_path,
        document_config_params=document_config_params,
        metadata_file=args.metadata_file,
        upload=args.upload,
        repo_id=args.repo_id,
        preprocess_text=args.preprocess_text,
        data_type=args.data_type,
        enable_language_filtering=args.enable_language_filtering,
        language_filter_threshold=args.language_filter_threshold,
        tokenizer_name=args.tokenizer_name,
    )


if __name__ == "__main__":
    main()
