# BabyLM Multilingual Dataset Pipeline

A modular pipeline for processing various data sources into standardized BabyLM multilingual datasets.

## Overview

This pipeline provides a flexible framework for:
1. Processing data from various sources (OpenSubtitles, CHILDES, educational content, etc.)
2. Converting to standardized BabyLM format
3. Uploading to HuggingFace Hub

## Project Structure

```
├── opensubtitles_processor.py   # OpenSubtitles-specific processing
├── babylm_dataset_builder.py    # General BabyLM dataset builder
├── hf_uploader.py              # HuggingFace upload utilities
├── main_pipeline.py            # Main entry point
├── requirements.txt            # Python dependencies
└── example_usage.sh           # Usage examples
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your HuggingFace token:
```
HF_TOKEN=your_huggingface_token_here
```

## Dataset Format

All datasets follow the BabyLM standard format:

| Column | Description | Example Values |
|--------|-------------|----------------|
| text | Document text | "this is sample text..." |
| category | Content type | child-directed-speech, educational, subtitles, etc. |
| data-source | Original source | OpenSubtitles, CHILDES, etc. |
| script | Writing system | latin, cyrillic, arabic, etc. |
| age-estimate | Target age | "4", "12-17", "n/a" |
| license | Data license | cc-by, cc-by-sa, etc. |
| misc | Additional metadata | JSON string with extra info |

## Usage

### Processing OpenSubtitles Data

```bash
python main_pipeline.py opensubtitles \
    --language afr \
    --script latin \
    --age-estimate "n/a" \
    --license "cc-by" \
    --batch-size 100 \
    --upload \
    --repo-id "username/babylm-afr"
```

### Processing Custom Sources

1. Prepare your text files in a directory
2. Optionally create a metadata JSON file mapping document IDs to metadata
3. Run the pipeline:

```bash
python main_pipeline.py custom \
    --language nld \
    --script latin \
    --age-estimate "2-5" \
    --license "cc-by-sa" \
    --data-source "CHILDES-Dutch" \
    --category "child-directed-speech" \
    --texts-dir "./your_texts" \
    --metadata-file "./metadata.json" \
    --upload \
    --repo-id "username/babylm-nld"
```

## Module Details

### opensubtitles_processor.py

Handles OpenSubtitles-specific functionality:
- Downloads language-specific zip files
- Extracts XML metadata from files
- Processes XML to clean text
- Generates file metadata CSV with all XML metadata fields

### babylm_dataset_builder.py

General-purpose dataset builder:
- Creates standardized BabyLM dataset structure
- Handles metadata management
- Generates final dataset tables (CSV/Parquet)
- Supports various content categories

### hf_uploader.py

HuggingFace integration:
- Uploads datasets to HuggingFace Hub
- Creates dataset cards automatically
- Handles authentication via environment variables

## Adding New Data Sources

To add support for a new data source:

1. Create a processor module similar to `opensubtitles_processor.py`
2. Implement text extraction and metadata handling
3. Add a new option in `main_pipeline.py`
4. Use `BabyLMDatasetBuilder` to create the final dataset

## Output Structure

Each processed dataset creates:
```
babylm-{language}/
├── texts/                    # Individual text files
├── dataset_metadata.json     # Complete metadata
├── {language}_file_metadata.csv  # File-level metadata (OpenSubtitles)
├── babylm-{language}_dataset.csv     # Final dataset (CSV)
├── babylm-{language}_dataset.parquet # Final dataset (Parquet)
└── README.md                # Dataset card
```

## Categories

Valid categories for BabyLM datasets:
- `child-directed-speech`: Direct speech to children
- `educational`: Educational content for children
- `child-books`: Children's literature
- `child-wiki`: Child-friendly encyclopedic content
- `child-news`: News adapted for children
- `subtitles`: TV/movie subtitles
- `qed`: QED educational content
- `child-available-speech`: Other dialogue/speech accessible to children

## License

The pipeline code is provided as-is. Individual datasets have their own licenses as specified in the metadata.