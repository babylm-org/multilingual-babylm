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
from text_preprocessor import SubtitlePreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Process OpenSubtitles data into BabyLM format"
    )
    
    # Required arguments
    parser.add_argument("--language", "-l", required=True,
                       help="ISO 639-3 language code")
    
    # OpenSubtitles specific
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing XML files")
    parser.add_argument("--keep-zip", action="store_true",
                       help="Keep the downloaded zip file")
    
    # Dataset configuration (with OpenSubtitles defaults)
    parser.add_argument("--script", required=True,
                       help="Script type (latin, cyrillic, arabic, etc.)")
    parser.add_argument("--age-estimate", default="n/a",
                       help="Age estimate (default: n/a for subtitles)")
    parser.add_argument("--license", default="cc-by",
                       help="License (default: cc-by)")
    
    # Upload options
    parser.add_argument("--upload", action="store_true",
                       help="Upload to HuggingFace after processing")
    parser.add_argument("--repo-id",
                       help="HuggingFace repo ID")
    
    # Preprocessing options
    parser.add_argument("--no-preprocess", action="store_true",
                       help="Skip preprocessing (use raw extracted text)")
    parser.add_argument("--lowercase", action="store_true", default=True,
                       help="Lowercase text (default: True)")
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false",
                       help="Don't lowercase text")
    
    # Optional metadata
    parser.add_argument("--source-url", default="https://opus.nlpl.eu/OpenSubtitles.php",
                       help="Source URL")
    parser.add_argument("--misc", type=json.loads,
                       help="Additional metadata as JSON string")
    
    args = parser.parse_args()
    
    # Step 1: Download and extract OpenSubtitles data
    print(f"Processing OpenSubtitles data for {args.language}...")
    
    # Create preprocessor for OpenSubtitles
    preprocessor = None
    if not args.no_preprocess:
        preprocessor = SubtitlePreprocessor(
            lowercase=args.lowercase,
            normalize_whitespace=True,
            fix_unicode=True,
            remove_timestamps=True,
            remove_stage_directions=True
        )
    
    # Process with OpenSubtitles processor
    processor = OpenSubtitlesProcessor(args.language)
    metadata_df, preprocessed_dir = processor.process_language(
        batch_size=args.batch_size,
        keep_zip=args.keep_zip,
        preprocessor=preprocessor
    )
    
    print(f"\nExtracted {len(metadata_df)} files to {preprocessed_dir}")
    
    # Create metadata mapping from the OpenSubtitles metadata
    metadata_mapping = {}
    for _, row in metadata_df.iterrows():
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
    
    # Save metadata mapping
    metadata_file = processor.output_dir / f"{args.language}_metadata_mapping.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata_mapping, f, indent=2)
    
    # Step 2: Use main pipeline to create BabyLM dataset
    print("\nCreating BabyLM dataset...")
    
    # Build command for main pipeline
    cmd = [
        sys.executable, "main_pipeline.py",
        "--language", args.language,
        "--data-source", "OpenSubtitles",
        "--category", "subtitles",
        "--texts-dir", str(preprocessed_dir),
        "--script", args.script,
        "--age-estimate", args.age_estimate,
        "--license", args.license,
        "--metadata-file", str(metadata_file),
        "--source-url", args.source_url
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
    
    # Run main pipeline
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running main pipeline: {e}")
        sys.exit(1)
    
    print("\nOpenSubtitles processing complete!")


if __name__ == "__main__":
    main()