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
from language_filter import filter_dataset_for_lang_and_script, update_dataset_scripts
from language_scripts import validate_script_code
from loader import get_loader
from pad_dataset import pad_dataset_to_next_tier, remove_padding_data
from multilingual_res.manager import fetch_resource, remove_resource

from iso639 import is_language, Lang

from loguru import logger
from logging_utils import setup_logger


def process_dataset(
    language_code: str,
    script_code: str,
    data_path: Optional[Path],
    document_config_params: dict,
    metadata_file: Optional[Path],
    upload: bool,
    repo_id: Optional[str],
    preprocess_text: bool,
    data_type: Optional[str],
    enable_language_filtering: bool,
    language_filter_threshold: float,
    pad_opensubtitles: bool,
    tokenizer_name: Optional[str],
    enable_script_update: bool = False,
    script_update_all: bool = False,
    overwrite: bool = False,
    add_ririro_data: bool = False,
    add_glotstorybook_data: bool = False,
    add_childwiki_data: bool = False,
    add_childes_data: bool = False,
    remove_previous_ririro_data: bool = False,
    remove_previous_glotstorybook_data: bool = False,
    remove_previous_childwiki_data: bool = False,
    remove_previous_childes_data: bool = False,
    remove_previous_padding: bool = False,
    byte_premium_factor: Optional[float] = None,
    create_pr: bool = False,
    pr_title="Update Dataset",
    pr_description="",
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
        pad_opensubtitles: Whether to pad dataset with OpenSubtitles
        tokenizer_name: Name of the tokenizer to use for token counting (for languages like Chinese, Japanese and Korean)
        overwrite: Whether to overwrite existing dataset instead of merging
        add_ririro_data: Whether to add Ririro resource for the language
        add_glotstorybook_data: Whether to add GlotStoryBook resource for the language
        add_childwiki_data: Whether to add ChildWiki resource for the language

    Returns:
        Path to output directory
    """

    logger.info(f"Processing data for {language_code}...")

    docs = []
    # 0. Load data using loader if both data_path and data_type are provided
    if data_path is not None and data_type is not None:
        loader = get_loader(data_type)
        docs.extend(loader.load_data(data_path))

    # 0.5 Remove previously added resources if requested
    if remove_previous_ririro_data:
        logger.info(
            f"Removing previously added Ririro resource for language: {language_code}"
        )
        docs = remove_resource("ririro", docs, language_code, script_code)

    if remove_previous_glotstorybook_data:
        logger.info(
            f"Removing previously added GlotStoryBook resource for language: {language_code}"
        )
        docs = remove_resource("glotstorybook", docs, language_code, script_code)

    if remove_previous_childwiki_data:
        logger.info(
            f"Removing previously added ChildWiki resource for language: {language_code}"
        )
        docs = remove_resource("childwiki", docs, language_code, script_code)

    if remove_previous_childes_data:
        logger.info(
            f"Removing previously added Childes resource for language: {language_code}"
        )
        docs = remove_resource("childes", docs, language_code, script_code)

    # remove padding data if requested
    if remove_previous_padding:
        docs = remove_padding_data(docs)

    # 1.0 Optionally fetch Ririro resource
    if add_ririro_data:
        logger.info(f"Fetching Ririro resource for language: {language_code}")
        ririro_docs = fetch_resource("ririro", language_code, script_code)
        docs.extend(ririro_docs)

    # 1.1 Optionally fetch GlotStoryBook resource
    if add_glotstorybook_data:
        logger.info(f"Fetching GlotStoryBook resource for language: {language_code}")
        glotstorybook_docs = fetch_resource("glotstorybook", language_code, script_code)
        docs.extend(glotstorybook_docs)

    # 1.2 Optionally fetch ChildWiki resource
    if add_childwiki_data:
        logger.info(f"Fetching ChildWiki resource for language: {language_code}")
        childwiki_docs = fetch_resource("childwiki", language_code, script_code)
        docs.extend(childwiki_docs)

    # 1.3 Optionally fetch Childes resource
    if add_childes_data:
        logger.info(f"Fetching Childes resource for language: {language_code}")
        childes_docs = fetch_resource("childes", language_code, script_code)
        docs.extend(childes_docs)

    if len(docs) == 0:
        logger.info(
            "No documents found. Please provide valid data_path and/or add multilingual resources. Aborting ..."
        )
        return Path()

    logger.info(f"Loaded {len(docs)} documents from data source(s)")

    # 2. Load metadata file if provided and merge
    metadata_mapping = {}
    if metadata_file and metadata_file.exists():
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_mapping = json.load(f)
    # Metadata file overrides document-level metadata
    for doc in docs:
        # Use file_name for mapping if present, else canonical 'doc-id'
        meta_key = doc.get("file_name") or doc.get("doc-id")
        if meta_key in metadata_mapping:
            doc["metadata"].update(metadata_mapping[meta_key])
    # Remove 'file_name' field before passing to builder
    for doc in docs:
        if "file_name" in doc:
            del doc["file_name"]

    # 3. Build dataset
    logger.info("Building dataset...")
    dataset_config = DatasetConfig(language_code=language_code)
    # Zzzz : default ISO 15924 value for Unknown or Unencoded
    builder = BabyLMDatasetBuilder(dataset_config, merge_existing=not overwrite)
    builder.add_documents_from_iterable(docs, document_config_params)
    builder.create_dataset_table()
    assert builder.dataset_table is not None

    # 4. Preprocess all texts (if requested)
    if preprocess_text:
        logger.info("Preprocessing document texts...")
    assert builder.dataset_table is not None
    builder.dataset_table = preprocess_dataset(builder.dataset_table)

    # 5. Language filtering if enabled
    if enable_language_filtering:
        logger.info(
            f"Filtering dataset for language {language_code} and script {script_code}..."
        )
        assert builder.dataset_table is not None
        builder.dataset_table = filter_dataset_for_lang_and_script(
            builder.dataset_table,
            language_code=language_code,
            script_code=script_code,
            language_filter_threshold=language_filter_threshold,
        )

    # 6. Pad dataset to next tier, accounting for byte premium
    if pad_opensubtitles:
        logger.info(f"Padding dataset for {language_code} using OpenSubtitles...")
        assert builder.dataset_table is not None
        results = pad_dataset_to_next_tier(
            dataset_df=builder.dataset_table,
            language_code=language_code,
            script_code=script_code,
            byte_premium_factor=byte_premium_factor,
        )
        builder.dataset_table = results["dataset"]
        # Keep the byte premium factor and dataset size for metadata
        setattr(builder, "byte_premium_factor", results["byte_premium_factor"])  # type: ignore[attr-defined]
        setattr(builder, "dataset_size", results["dataset_size"])  # type: ignore[attr-defined]

        # assume the padding dataset is filtered for language and script
        # and has been preprocessed for the subtitles category

    # 6.5. Deduplicate by exact text before saving
    logger.info("Running deduplication on dataset before saving...")
    builder.deduplicate_by_text()

    # 6.7 Update scripts (optional; default disabled). Default scope: newly added docs only.
    if enable_script_update:
        assert builder.dataset_table is not None
        if script_update_all:
            mask = builder.dataset_table["doc-id"].astype(str).notna()
            scope_msg = "all documents"
        else:
            if hasattr(builder, "_existing_doc_ids") and isinstance(
                builder._existing_doc_ids, set
            ):
                mask = ~builder.dataset_table["doc-id"].astype(str).isin(
                    builder._existing_doc_ids
                )
            else:
                # If we don't have existing ids (fresh dataset), treat all rows as new
                mask = builder.dataset_table["doc-id"].astype(str).notna()
            scope_msg = "newly added (incl. padding) documents only"

        if mask.any():
            logger.info(f"Updating script annotations for {scope_msg}...")
            updated_subset = update_dataset_scripts(
                builder.dataset_table.loc[mask].copy()
            )
            builder.dataset_table.loc[mask, "script"] = updated_subset["script"].values

    # 7. Save and create dataset
    builder.save_dataset()
    assert builder.dataset_table is not None
    logger.info(f"\nDataset created with {len(builder.dataset_table)} documents")

    # 8. Upload if requested
    if upload and repo_id:
        logger.info(f"\nUploading to HuggingFace: {repo_id}")
        uploader = HFDatasetUploader()
        uploader.upload_babylm_dataset(
            language_code=language_code,
            script_code=script_code,
            dataset_dir=builder.output_dir,
            repo_id=repo_id,
            create_repo_if_missing=True,
            tokenizer_name=tokenizer_name,
            byte_premium_factor=byte_premium_factor,
            create_pr=create_pr,
            pr_title=pr_title,
            pr_description=pr_description,
        )

    return builder.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Process data into BabyLM format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request",
    )
    parser.add_argument(
        "--pr-title",
        type=str,
        default="Update Dataset",
        help="Pull Request Title",
    )
    parser.add_argument(
        "--pr-description",
        type=str,
        default="",
        help="Pull Request Title",
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
        default=None,
        help="Path to data directory or file",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default=None,
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
            "padding",
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
    # Script update toggle (default disabled)
    parser.add_argument(
        "--enable-script-update",
        dest="enable_script_update",
        action="store_true",
        help="Enable script identification and updates (default: disabled)",
    )
    parser.add_argument(
        "--script-update-all",
        dest="script_update_all",
        action="store_true",
        help="If set (and script update is enabled), update scripts for the whole dataset. By default, only newly added documents are updated.",
    )
    parser.add_argument(
        "--preprocess",
        "--preprocess-text",
        dest="preprocess_text",
        action="store_true",
        help="Enable text preprocessing",
    )
    parser.add_argument(
        "--pad",
        "--pad-opensubtitles",
        dest="pad_opensubtitles",
        action="store_true",
        help="Enable padding with OpenSubtitles, FineWeb-C, or Wikipedia (alias: --pad)",
    )
    parser.add_argument(
        "--remove-previous-padding",
        action="store_true",
        help="If set, remove previously added padding for the given language.",
    )
    parser.add_argument(
        "--byte-premium-factor",
        type=float,
        default=None,
        help="Provide byte-premium factor manually, instead of retrieving it automatically (override).",
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
    parser.add_argument(
        "--add-ririro-data",
        action="store_true",
        help="If set, fetch and add Ririro resource for the given language before processing other data.",
    )
    parser.add_argument(
        "--add-glotstorybook-data",
        action="store_true",
        help="If set, fetch and add GlotStoryBook resource for the given language before processing other data.",
    )
    parser.add_argument(
        "--add-childwiki-data",
        action="store_true",
        help="If set, fetch and add ChildWiki resource for the given language before processing other data.",
    )
    parser.add_argument(
        "--add-childes-data",
        action="store_true",
        help="If set, fetch and add Childes resource for the given language before processing other data.",
    )

    parser.add_argument(
        "--remove-previous-ririro-data",
        action="store_true",
        help="If set, remove Ririro previously added resource for the given language.",
    )
    parser.add_argument(
        "--remove-previous-glotstorybook-data",
        action="store_true",
        help="If set, remove GlotStoryBook previously added resource for the given language.",
    )
    parser.add_argument(
        "--remove-previous-childwiki-data",
        action="store_true",
        help="If set, remove ChildWiki previously added resource for the given language.",
    )
    parser.add_argument(
        "--remove-previous-childes-data",
        action="store_true",
        help="If set, remove Childes previously added resource for the given language.",
    )

    parser.add_argument(
        "--logfile",
        type=str,
        help="logging filepath",
        default="logs/log_pipeline.txt"
    )


    args = parser.parse_args()

    if not validate_script_code(args.script):
        raise ValueError(
            f"Invalid script code '{args.script}'. Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)"
        )

    if is_language is not None and not is_language(args.language, "pt3"):
        if Lang is not None and is_language(Lang(args.language).pt3, "pt3"):
            args.language = Lang(args.language).pt3  # Normalize to ISO 639-3 if needed
        else:
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

    
    setup_logger(args.logfile)

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
        enable_script_update=args.enable_script_update,
        script_update_all=args.script_update_all,
        language_filter_threshold=args.language_filter_threshold,
        pad_opensubtitles=args.pad_opensubtitles,
        tokenizer_name=args.tokenizer_name,
        overwrite=args.overwrite,
        add_ririro_data=args.add_ririro_data,
        add_glotstorybook_data=args.add_glotstorybook_data,
        add_childwiki_data=args.add_childwiki_data,
        add_childes_data=args.add_childes_data,
        remove_previous_ririro_data=args.remove_previous_ririro_data,
        remove_previous_glotstorybook_data=args.remove_previous_glotstorybook_data,
        remove_previous_childwiki_data=args.remove_previous_childwiki_data,
        remove_previous_childes_data=args.remove_previous_childes_data,
        remove_previous_padding=args.remove_previous_padding,
        byte_premium_factor=args.byte_premium_factor,
        create_pr=args.create_pr,
        pr_description=args.pr_description,
        pr_title=args.pr_title,
    )


if __name__ == "__main__":
    main()
