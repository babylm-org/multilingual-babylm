"""
Specific script for processing OpenSubtitles data.
This wraps the OpenSubtitles processor and main pipeline for convenience.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

from opensubtitles_processor import OpenSubtitlesProcessor
from preprocessor import SubtitlePreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Process OpenSubtitles data into BabyLM format"
    )

    # Required arguments
    parser.add_argument(
        "--language", "-l", required=True, help="ISO 639-3 language code"
    )

    # OpenSubtitles specific
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing XML files"
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="Keep the downloaded zip file"
    )

    # IMDB filtering options
    parser.add_argument(
        "--imdb-db-path",
        type=Path,
        default=Path("./prep_subtitles/imdb_mastersheet.db"),
        help="Path to IMDB SQLite database",
    )
    parser.add_argument(
        "--forbidden-genres",
        nargs="+",
        default=[],
        help="List of genres to exclude (e.g., --forbidden-genres Horror News)",
    )
    parser.add_argument(
        "--disable-imdb-filtering",
        action="store_true",
        help="Disable IMDB-based filtering",
    )

    # Dataset configuration (with OpenSubtitles defaults)
    parser.add_argument(
        "--script", required=True, help="Script type (latin, cyrillic, arabic, etc.)"
    )
    parser.add_argument(
        "--age-estimate",
        default="n/a",
        help="Age estimate (default: n/a for subtitles)",
    )
    parser.add_argument("--license", default="cc-by", help="License (default: cc-by)")

    # Upload options
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace after processing"
    )
    parser.add_argument("--repo-id", help="HuggingFace repo ID")

    # Preprocessing options
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing (use raw extracted text)",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=False,
        help="Lowercase text (default: False - preserves capitalization)",
    )
    parser.add_argument(
        "--no-lowercase",
        dest="lowercase",
        action="store_false",
        help="Don't lowercase text (default behavior)",
    )

    # Optional metadata
    parser.add_argument(
        "--source-url",
        default="https://opus.nlpl.eu/OpenSubtitles.php",
        help="Source URL",
    )
    parser.add_argument(
        "--misc", type=json.loads, help="Additional metadata as JSON string"
    )

    args = parser.parse_args()

    # Validate IMDB database if filtering is enabled
    enable_imdb_filtering = not args.disable_imdb_filtering
    if enable_imdb_filtering and not args.imdb_db_path.exists():
        print(f"Warning: IMDB database not found at {args.imdb_db_path}")
        print("Proceeding without IMDB filtering...")
        enable_imdb_filtering = False

    # Step 1: Download and extract OpenSubtitles data
    print(f"Processing OpenSubtitles data for {args.language}...")

    if enable_imdb_filtering:
        print("IMDB filtering enabled:")
        print(f"  Database: {args.imdb_db_path}")
        print(f"  Forbidden genres: {args.forbidden_genres}")
        print("  Adult content: excluded")
    else:
        print("IMDB filtering disabled")

    # Create preprocessor for OpenSubtitles
    preprocessor = None
    if not args.no_preprocess:
        preprocessor = SubtitlePreprocessor()

    # Process with OpenSubtitles processor
    processor = OpenSubtitlesProcessor(
        lang_code=args.language,
        imdb_db_path=args.imdb_db_path,
        forbidden_genres=args.forbidden_genres,
    )

    metadata_df, preprocessed_dir = processor.process_language(
        batch_size=args.batch_size,
        keep_zip=args.keep_zip,
        preprocessor=preprocessor,
        enable_imdb_filtering=enable_imdb_filtering,
    )

    print(f"\nExtracted {len(metadata_df)} files to {preprocessed_dir}")

    # Print filtering statistics
    if enable_imdb_filtering:
        success_count = len(metadata_df[metadata_df["processing_status"] == "success"])
        filtered_count = len(
            metadata_df[metadata_df["processing_status"] == "filtered_out"]
        )
        failed_count = len(metadata_df[metadata_df["processing_status"] == "failed"])

        print("\nProcessing Summary:")
        print(f"  Successfully processed: {success_count}")
        print(f"  Filtered out by IMDB criteria: {filtered_count}")
        print(f"  Failed to process: {failed_count}")

    # Create metadata mapping from the OpenSubtitles metadata (only successful ones)
    metadata_mapping = {}
    successful_df = metadata_df[metadata_df["processing_status"] == "success"]

    for _, row in successful_df.iterrows():
        doc_metadata = {"year": row["year"], "folder_name": row["folder_name"]}
        # Add any XML metadata fields
        for col in row.index:
            if col.startswith("meta_"):
                doc_metadata[col] = row[col]
        metadata_mapping[row["file_id"]] = doc_metadata

    # Save metadata mapping
    metadata_file = processor.output_dir / f"{args.language}_metadata_mapping.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_mapping, f, indent=2)

    # Step 2: Use new pipeline to create BabyLM dataset
    print("\nCreating BabyLM dataset with new pipeline...")

    # Build command for pipeline.py
    cmd = [
        sys.executable,
        "pipeline.py",
        "--language",
        args.language,
        "--script",
        args.script,
        "--data-path",
        str(preprocessed_dir),
        "--data-type",
        "text",
        "--data-source",
        "OpenSubtitles",
        "--category",
        "subtitles",
        "--age-estimate",
        args.age_estimate,
        "--license",
        args.license,
        "--metadata-file",
        str(metadata_file),
    ]

    # Add misc metadata if provided
    if args.misc:
        cmd.extend(["--misc", json.dumps(args.misc)])

    # Add upload options if specified
    if args.upload:
        cmd.append("--upload")
        if args.repo_id:
            cmd.extend(["--repo-id", args.repo_id])
        else:
            print("Error: --repo-id required when --upload is specified")
            sys.exit(1)

    # Run pipeline
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)

    print("\nOpenSubtitles processing complete!")


if __name__ == "__main__":
    main()
