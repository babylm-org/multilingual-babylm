# BabyLM Multilingual Dataset Pipeline

A modular pipeline for processing various data sources into standardized BabyLM multilingual datasets.

## Overview

This pipeline provides a flexible framework for:
1. Processing text data from any source into BabyLM format
2. Applying various preprocessing strategies (including LLM-based filtering)
3. Uploading to HuggingFace Hub

## Project Structure

```
├── main_pipeline.py             # Main generic pipeline
├── babylm_dataset_builder.py    # BabyLM dataset builder
├── text_preprocessor.py         # Text preprocessing utilities
├── hf_uploader.py              # HuggingFace upload utilities
├── process_opensubtitles.py    # OpenSubtitles-specific wrapper
├── opensubtitles_processor.py  # OpenSubtitles processing module
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

### Basic Usage (Any Text Source)

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MyDataSource" \
    --category "educational" \
    --texts-dir "./path/to/texts" \
    --script latin \
    --age-estimate "6-12" \
    --license "cc-by"
```

### With Preprocessing

The pipeline supports multiple preprocessing strategies:

#### Basic Text Preprocessing
```bash
python main_pipeline.py \
    --language fra \
    --data-source "FrenchTexts" \
    --category "child-books" \
    --texts-dir "./texts" \
    --script latin \
    --age-estimate "4-8" \
    --license "cc-by" \
    --preprocess \
    --lowercase \
    --fix-unicode
```

#### Subtitle-Specific Preprocessing
```bash
python main_pipeline.py \
    --language deu \
    --data-source "GermanSubtitles" \
    --category "subtitles" \
    --texts-dir "./subtitles" \
    --script latin \
    --age-estimate "n/a" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type subtitle \
    --remove-timestamps \
    --remove-stage-directions
```

#### LLM-Based Filtering
```bash
python main_pipeline.py \
    --language spa \
    --data-source "WebTexts" \
    --category "child-available-speech" \
    --texts-dir "./web_texts" \
    --script latin \
    --age-estimate "8-14" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type llm \
    --llm-model "llama3.2" \
    --llm-prompt "Your custom filtering prompt here" \
    --llm-filter-threshold 0.8
```

### Processing OpenSubtitles Data

For OpenSubtitles specifically, use the dedicated wrapper script:

```bash
python process_opensubtitles.py \
    --language af \
    --script latin \
    --batch-size 100 \
    --upload \
    --repo-id "bhargavns/babylm-afr"
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
