# main_pipeline.py
"""
Main pipeline script for processing various data sources into BabyLM datasets.
"""

import argparse
import json
from pathlib import Path

# Import our modules
from babylm_dataset_builder import BabyLMDatasetBuilder, DatasetConfig
from hf_uploader import HFDatasetUploader
from language_filter import LanguageFilter, print_filtering_results
from language_scripts import SCRIPT_NAMES
from text_preprocessor import apply_llm_preprocessing, preprocess_dataset


def validate_args(parser):
    args = parser.parse_args()
    # Normalize script code to title case
    args.script = args.script.title()

    # Enforce ISO 15924 script code
    if args.script not in SCRIPT_NAMES:
        parser.error(
            f"Invalid script code '{args.script}'. Please use a valid ISO 15924 script code (e.g., Latn, Cyrl, Arab, etc.)."
        )

    if args.upload and not args.repo_id:
        parser.error("--repo-id is required when --upload is specified")

    return args


def process_dataset(
    language_code: str,
    script_code: str,
    load_path: Path,
    load_format: str,
    document_config_params: dict,
    metadata_file: Path | None,
    upload: bool,
    repo_id: str | None,
    preprocess_text: bool,
    enable_language_filtering: bool,
    language_filter_threshold: float,
    tokenizer_name: str | None,
    hf_dataset_split: str | None = None,
    llm_preprocessing_args: dict | None = None,
) -> Path:
    """
    Process any data source into BabyLM format.

    Args:
        language_code: ISO 639-3 language code
        script_code: ISO 15924 script code (e.g., Latn, Cyrl, Arab)
        load_path: Directory containing data to process
        load_format: Format of the input (text, json, jsonl, csv, hf)
        document_config_params: Dictionary with default document-level configuration,
            is overwritten by document-level metadata
        metadata_file: Optional JSON file with document metadata
        upload: Whether to upload to HuggingFace
        repo_id: HuggingFace repository ID
        preprocess_text: Whether to preprocess the text
        enable_language_filtering: Whether to enable language filtering
        language_filter_threshold: Minimum confidence for language filtering
        tokenizer_name: Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)
        hf_dataset_split: Optional split name for HuggingFace datasets (e.g., 'train', 'test')

    Returns:
        Path to output directory

    """
    # create dataset builder
    dataset_config = DatasetConfig(language_code=language_code)
    builder = BabyLMDatasetBuilder(dataset_config)

    # load documents
    if load_format == "text":
        with open(metadata_file, encoding="utf-8") as f:
            metadata_mapping = json.load(f)
        builder.add_documents_from_text_directory(
            load_path, document_config_params, metadata_mapping
        )
    else:
        builder.add_documents_from_path(
            load_path,
            document_config_params,
            hf_dataset_split=hf_dataset_split,
        )

    builder.create_dataset_table()

    # Language filtering if enabled
    if enable_language_filtering:
        expected_script = script_code
        print("\nPerforming language filtering...")
        print(f"Expected language: {language_code}")
        print(f"Expected script: {expected_script}")

        # Create language filter
        language_filter = LanguageFilter()

        # Perform filtering
        filter_results = language_filter.filter_documents(
            documennts_df=builder.dataset_table,
            expected_language=language_code,
            expected_script=expected_script,
            min_confidence=language_filter_threshold,
        )

        # keep only matching documents
        match_indexes = filter_results["match_indexes"]
        builder.dataset_table = builder.dataset_table.iloc[match_indexes]

        # Print filtering results
        print_filtering_results(filter_results, language_code, expected_script)

    # Preprocess documents if enabled
    if preprocess_text:
        builder.dataset_table = preprocess_dataset(builder.dataset_table)
        # Optionally apply LLM preprocessing afterwards
        if llm_preprocessing_args:
            builder.dataset_table = apply_llm_preprocessing(
                builder.dataset_table, llm_preprocessing_args
            )
            # # re-apply preprocessing to ensure consistency after LLM processing
            # builder.dataset_table = preprocess_dataset(builder.dataset_table)

    # Save and create dataset
    builder.save_dataset()
    print(f"\nDataset created with {len(builder.dataset_table)} documents")

    # Upload if requested
    if upload and repo_id:
        print(f"\nUploading to HuggingFace: {repo_id}")
        uploader = HFDatasetUploader()
        if tokenizer_name is not None:
            uploader.upload_babylm_dataset(
                dataset_dir=builder.output_dir,
                repo_id=repo_id,
                create_repo_if_missing=True,
                tokenizer_name=tokenizer_name,
            )
        else:
            uploader.upload_babylm_dataset(
                dataset_dir=builder.output_dir,
                repo_id=repo_id,
                create_repo_if_missing=True,
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
        "--script", required=True, help="Script type (latin, cyrillic, arabic, etc.)"
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        help="Directory containing text files to process",
        required=False,
    )
    parser.add_argument(
        "--load-format",
        type=str,
        choices=["text", "json", "jsonl", "csv", "hf"],
    )

    # Input file for txt documents
    parser.add_argument(
        "--metadata-file", type=Path, help="JSON file with document metadata"
    )

    # Default config parameters
    parser.add_argument(
        "--category",
        type=str,
    )
    parser.add_argument("--data-source", type=str)
    parser.add_argument(
        "--age-estimate",
        type=str,
    )
    parser.add_argument(
        "--license",
        type=str,
    )
    parser.add_argument(
        "--misc",
        type=str,
    )

    # Upload arguments
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace after processing",
        default=False,
    )
    parser.add_argument(
        "--repo-id",
        help="HuggingFace repo ID (e.g., 'username/babylm-eng')",
        type=str,
        default=None,
    )

    # Language filtering arguments
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

    # HF-specific arguments
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default=None,
        help="Split name for HuggingFace datasets (e.g., 'train', 'test'). Default: None (use default split)",
    )

    # Tokenizer options
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)",
    )

    # Text preprocessing options
    parser.add_argument(
        "--preprocess-text", action="store_true", help="Enable text preprocessing"
    )

    # LLM preprocessing options
    parser.add_argument(
        "--llm-preprocessing",
        action="store_true",
        help="Enable LLM preprocessing for text filtering/processing",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2",
        help="Ollama model to use for LLM preprocessing",
    )
    parser.add_argument("--llm-prompt", help="Prompt for LLM filtering/processing")
    parser.add_argument(
        "--llm-filter-threshold",
        type=float,
        default=0.7,
        help="Threshold for LLM filtering (0-1)",
    )
    parser.add_argument(
        "--llm-process-lines",
        type=bool,
        default=False,
        help="Whether to process text in chunks (line by line)",
    )

    args = validate_args(parser)

    # Prepare document config parameters
    document_config_params = {"script": args.script}
    if args.age_estimate:
        document_config_params["age_estimate"] = args.age_estimate
    if args.license:
        document_config_params["license"] = args.license
    if args.misc:
        document_config_params["license"] = args.license
    if args.data_source:
        document_config_params["data_source"] = args.data_source
    if args.category:
        document_config_params["category"] = args.category

    # Prepare LLM preprocessing arguments if enabled
    if args.llm_preprocessing:
        llm_preprocessing_args = {
            "model": args.llm_model,
            "prompt": args.llm_prompt,
            "filter_threshold": args.llm_filter_threshold,
            "process_lines": args.llm_process_lines,
        }
    else:
        llm_preprocessing_args = None

    # Process the dataset
    output_dir = process_dataset(
        language_code=args.language,
        script_code=args.script,
        load_path=args.load_path,
        load_format=args.load_format,
        document_config_params=document_config_params,
        metadata_file=args.metadata_file,
        upload=args.upload,
        repo_id=args.repo_id,
        preprocess_text=args.preprocess_text,
        enable_language_filtering=args.enable_language_filtering,
        language_filter_threshold=args.language_filter_threshold,
        tokenizer_name=args.tokenizer_name,
        hf_dataset_split=args.hf_dataset_split,
        llm_preprocessing_args=llm_preprocessing_args,
    )

    print(f"\nProcessing complete! Output directory: {output_dir}")


if __name__ == "__main__":
    main()
