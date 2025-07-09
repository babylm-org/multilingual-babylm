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
from language_filter import filter_dataset_for_lang_and_script
from language_scripts import validate_script_code
from loader import get_loader
from pad_dataset import pad_dataset_to_next_tier

from iso639 import is_language


def process_dataset(
    language_code: str,
    script_code: str,
    data_path: Path,
    document_config_params: dict,
    metadata_file: Optional[Path],
    upload: bool,
    repo_id: Optional[str],
    preprocess_text: bool,
    data_type: str,
    enable_language_filtering: bool,
    language_filter_threshold: float,
    pad_opensubtitles: bool,
    tokenizer_name: Optional[str],
    overwrite: bool = False,
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
        # Use file_name for mapping if present, else doc_id
        meta_key = doc.get("file_name") or doc["doc_id"]
        if meta_key in metadata_mapping:
            doc["metadata"].update(metadata_mapping[meta_key])
    # Remove 'file_name' field before passing to builder
    for doc in docs:
        if "file_name" in doc:
            del doc["file_name"]

    # 3. Build dataset
    dataset_config = DatasetConfig(language_code=language_code)
    # Zzzz : default ISO 15924 value for Unknown or Unencoded
    builder = BabyLMDatasetBuilder(dataset_config, merge_existing=not overwrite)
    builder.add_documents_from_iterable(docs, document_config_params)
    builder.create_dataset_table()

    # 4. Preprocess all texts (if requested)
    if preprocess_text:
        builder.dataset_table = preprocess_dataset(builder.dataset_table)

    # 5. Language filtering if enabled
    if enable_language_filtering:
        builder.dataset_table = filter_dataset_for_lang_and_script(
            builder.dataset_table,
            language_code=language_code,
            script_code=script_code,
            language_filter_threshold=language_filter_threshold,
        )

    # 6. Pad dataset to next tier, accounting for byte premium
    if pad_opensubtitles:
        results = pad_dataset_to_next_tier(
            dataset_df=builder.dataset_table,
            language_code=language_code,
        )
        builder.dataset_table = results["dataset"]
        # Keep the byte premium factor and dataset size for metadata
        builder.byte_premium_factor = results["byte_premium_factor"]
        builder.dataset_size = results["dataset_size"]

        # assume the padding dataset is filtered for language and script
        # and has been preprocessed for the subtitles category

    # 7. Save and create dataset
    builder.save_dataset()
    print(f"\nDataset created with {len(builder.dataset_table)} documents")

    # 8. Upload if requested
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
        "--script",
        required=True,
        help="Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)",
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
    parser.add_argument("--license", help="License (e.g., cc-by, cc-by-sa)")
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
        "--pad-opensubtitles",
        action="store_true",
        help="Enable padding with OpenSubtitles",
    )

    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset instead of merging",
    )

    args = parser.parse_args()

    if not validate_script_code(args.script):
        raise ValueError(
            f"Invalid script code '{args.script}'. Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)"
        )

    if not is_language(args.language, "pt3"):
        raise ValueError(
            f"Invalid language code '{args.language}'. Must be a valid ISO 639-3 code."
        )

    document_config_params = {
        "script": args.script,
    }
    if args.data_source:
        document_config_params["data-source"] = args.data_source
    if args.category:
        document_config_params["category"] = args.category
    if args.age_estimate:
        document_config_params["age-estimate"] = args.age_estimate
    if args.license:
        document_config_params["license"] = args.license
    if args.misc:
        document_config_params["misc"] = args.misc

    process_dataset(
        language_code=args.language,
        script_code=args.script,
        data_path=args.data_path,
        document_config_params=document_config_params,
        metadata_file=args.metadata_file,
        upload=args.upload,
        repo_id=args.repo_id,
        preprocess_text=args.preprocess_text,
        data_type=args.data_type,
        enable_language_filtering=args.enable_language_filtering,
        language_filter_threshold=args.language_filter_threshold,
        pad_opensubtitles=args.pad_opensubtitles,
        tokenizer_name=args.tokenizer_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
