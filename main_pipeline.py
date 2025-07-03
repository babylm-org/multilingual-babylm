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
from text_preprocessor import remove_urls, normalize_punctuation, remove_xml_tags
from language_filter import LanguageFilter, print_filtering_results
from language_scripts import get_script_formal_name, SCRIPT_NAMES


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
    preprocessor_type: str = "text",
    enable_language_filtering: bool = False,
    language_filter_threshold: float = 0.8,
    tokenizer_name: Optional[str] = None,
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
        enable_language_filtering: Whether to enable language filtering
        language_filter_threshold: Minimum confidence for language filtering
        tokenizer_name: Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)

    Returns:
        Path to output directory
    """
    print(f"Processing {data_source} data for {language_code}...")

    # Create preprocessor if preprocessing is requested
    preprocessor = None
    metadata_mapping = {}
    if preprocessing_config:
        print(f"Using {preprocessor_type} preprocessor...")
        preprocessor = create_preprocessor(preprocessor_type, **preprocessing_config)

        if preprocessor:
            preprocessed_dir = Path(f"./preprocessed_{data_source}_{language_code}")
            preprocessed_dir.mkdir(exist_ok=True)

            if preprocessor_type == "csv" and hasattr(preprocessor, "process_csv"):
                print(f"Preprocessing CSV file: {texts_dir}")
                # type: ignore
                metadata_mapping = getattr(preprocessor, "process_csv")(
                    texts_dir, preprocessed_dir
                )
                texts_dir = preprocessed_dir
            elif preprocessor_type == "hf" and hasattr(
                preprocessor, "process_hf_dataset"
            ):
                print(f"Preprocessing HuggingFace dataset: {texts_dir}")
                # type: ignore
                metadata_mapping = getattr(preprocessor, "process_hf_dataset")(
                    preprocessed_dir
                )
                texts_dir = preprocessed_dir
            elif preprocessor_type == "json" and hasattr(preprocessor, "process_json"):
                print(f"Preprocessing JSON file: {texts_dir}")
                # type: ignore
                metadata_mapping = getattr(preprocessor, "process_json")(
                    texts_dir, preprocessed_dir
                )
                texts_dir = preprocessed_dir

            print("Preprocessing text files...")
            text_files = list(texts_dir.glob("*.txt"))
            for text_file in tqdm(text_files, desc="Preprocessing files"):
                output_file = preprocessed_dir / text_file.name
                try:
                    preprocessor.process_file(text_file, output_file)
                except Exception as e:
                    print(f"Error preprocessing {text_file}: {e}")
                    continue
            texts_dir = preprocessed_dir  # Create dataset config (just language code)
    dataset_config = DatasetConfig(language_code=language_code)

    # Create default document config
    # Map script code to formal name for dataset
    script_code = document_config_params.get("script", "Zzzz")
    script_formal_name = get_script_formal_name(script_code)
    document_config_params["script"] = script_formal_name
    default_doc_config = DocumentConfig(
        category=category, data_source=data_source, **document_config_params
    )

    # Build dataset
    builder = BabyLMDatasetBuilder(dataset_config)

    # Language filtering if enabled
    if enable_language_filtering:
        expected_script = script_code
        print(f"\nPerforming language filtering...")
        print(f"Expected language: {language_code}")
        print(f"Expected script: {expected_script}")

        # Create language filter
        language_filter = LanguageFilter()

        # Create filtered directory inside builder output directory
        filtered_dir = builder.output_dir / "filtered"

        # Perform filtering
        filter_results = language_filter.filter_documents(
            input_dir=texts_dir,
            expected_language=language_code,
            expected_script=expected_script,
            output_dir=filtered_dir,
            min_confidence=language_filter_threshold,
        )

        # Print filtering results
        print_filtering_results(filter_results, language_code, expected_script)

        # Use matching files for dataset building
        matching_dir = filtered_dir / language_code
        if matching_dir.exists():
            texts_dir = matching_dir
        else:
            print(
                f"Warning: No matching files found for {language_code}_{expected_script}"
            )
            print(f"Proceeding with original directory: {texts_dir}")

    # Load metadata if provided, unless already set from preprocessor
    if not metadata_mapping and metadata_file and metadata_file.exists():
        print(f"Loading metadata from {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_mapping = json.load(f)
    # Validate script field in all loaded metadata
    if metadata_mapping:
        for doc_id, meta in metadata_mapping.items():
            if isinstance(meta, dict) and "script" in meta and meta["script"] not in SCRIPT_NAMES:
                raise ValueError(f"Invalid script code '{meta['script']}' in metadata for doc_id {doc_id}. Must be ISO 15924 code.")

    # Add documents
    builder.add_documents_from_directory(
        texts_dir, default_doc_config, metadata_mapping
    )

    # Save and create dataset
    builder.save_metadata()
    dataset_df = builder.create_dataset_table()
    print(f"\nDataset created with {len(dataset_df)} documents")

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
        "--data-source",
        "-s",
        required=True,
        help="Data source name (e.g., OpenSubtitles, CHILDES, etc.)",
    )
    parser.add_argument(
        "--category",
        "-c",
        required=True,
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
        "--texts-dir",
        "-t",
        type=Path,
        required=True,
        help="Directory containing text files to process",
    )

    # Dataset configuration
    parser.add_argument(
        "--script", required=True, help="Script type (latin, cyrillic, arabic, etc.)"
    )
    parser.add_argument(
        "--age-estimate", required=True, help="Age estimate (e.g., '4', '12-17', 'n/a')"
    )
    parser.add_argument(
        "--license", required=True, help="License (e.g., cc-by, cc-by-sa)"
    )

    # Optional metadata
    parser.add_argument(
        "--metadata-file", type=Path, help="JSON file with document metadata"
    )
    parser.add_argument("--source-url", help="Source URL")
    parser.add_argument("--source-identifier", help="Source identifier")
    parser.add_argument(
        "--misc", type=json.loads, help="Additional metadata as JSON string"
    )  # Upload arguments
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace after processing"
    )
    parser.add_argument(
        "--repo-id", help="HuggingFace repo ID (e.g., 'username/babylm-eng')"
    )  # Language filtering arguments
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

    # Preprocessing arguments
    parser.add_argument(
        "--preprocess", action="store_true", help="Enable text preprocessing"
    )
    parser.add_argument(
        "--preprocessor-type",
        default="text",
        choices=["text", "subtitle", "transcript", "llm", "csv", "hf", "json"],
        help="Type of preprocessor to use",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name for text in CSV or HuggingFace datasets (default: 'text')",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default=None,
        help="Split name for HuggingFace datasets (e.g., 'train', 'test'). Default: None (use default split)",
    )

    # Preprocessing options
    parser.add_argument(
        "--lowercase", action="store_true", default=None, help="Lowercase text"
    )
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="Don't lowercase text",
    )
    parser.add_argument(
        "--fix-unicode",
        action="store_true",
        default=None,
        help="Fix unicode issues with ftfy",
    )
    parser.add_argument(
        "--no-fix-unicode",
        dest="fix_unicode",
        action="store_false",
        help="Don't fix unicode",
    )
    parser.add_argument(
        "--remove-timestamps",
        action="store_true",
        default=None,
        help="Remove timestamps",
    )
    parser.add_argument(
        "--remove-stage-directions",
        action="store_true",
        default=None,
        help="Remove [bracketed] stage directions",
    )
    parser.add_argument(
        "--remove-urls", action="store_true", default=None, help="Remove URLs"
    )
    parser.add_argument(
        "--normalize-punctuation",
        action="store_true",
        default=None,
        help="Normalize punctuation",
    )
    parser.add_argument(
        "--remove-xml-tags", action="store_true", default=None, help="Remove XML tags"
    )
    parser.add_argument(
        "--replace-newline-within-paragraph",
        action="store_true",
        default=False,
        help="Replace single newlines with space within paragraphs (default: False)",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)",
    )

    # LLM preprocessing options
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

    args = parser.parse_args()

    # Normalize script code to title case
    args.script = args.script.title()
    # Enforce ISO 15924 script code
    if args.script not in SCRIPT_NAMES:
        parser.error(
            f"Invalid script code '{args.script}'. Please use a valid ISO 15924 script code (e.g., Latn, Cyrl, Arab, etc.)."
        )

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
        custom_steps = []
        # Add boolean options only if explicitly set
        if args.lowercase is not None:
            preprocessing_config["lowercase"] = args.lowercase
        if args.fix_unicode is not None:
            preprocessing_config["fix_unicode"] = args.fix_unicode
        if args.remove_timestamps is not None:
            preprocessing_config["remove_timestamps"] = args.remove_timestamps
        if args.remove_stage_directions is not None:
            preprocessing_config["remove_stage_directions"] = (
                args.remove_stage_directions
            )
        preprocessing_config["replace_newline_within_paragraph"] = (
            args.replace_newline_within_paragraph
        )
        # Add custom preprocessing steps if flags are set
        if args.remove_urls:
            custom_steps.append(remove_urls)
        if args.normalize_punctuation:
            custom_steps.append(normalize_punctuation)
        if args.remove_xml_tags:
            custom_steps.append(remove_xml_tags)
        if custom_steps:
            preprocessing_config["custom_steps"] = custom_steps
        # Add LLM-specific options if using LLM preprocessor
        if args.preprocessor_type == "llm":
            if not args.llm_prompt:
                parser.error("--llm-prompt is required when using LLM preprocessor")
            preprocessing_config["model"] = args.llm_model
            preprocessing_config["prompt"] = args.llm_prompt
            preprocessing_config["filter_threshold"] = args.llm_filter_threshold
        # Add CSV/HF/JSON-specific options
        if args.preprocessor_type in ["csv", "hf", "json"]:
            preprocessing_config["text_field"] = args.text_field
        if args.preprocessor_type == "hf":
            preprocessing_config["dataset_id"] = str(args.texts_dir)
            preprocessing_config["split"] = args.hf_dataset_split

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
        preprocessor_type=args.preprocessor_type,
        enable_language_filtering=args.enable_language_filtering,
        language_filter_threshold=args.language_filter_threshold,
        tokenizer_name=args.tokenizer_name,
    )

    print(f"\nProcessing complete! Output directory: {output_dir}")


if __name__ == "__main__":
    main()
