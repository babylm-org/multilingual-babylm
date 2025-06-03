# BabyLM Multilingual Dataset Pipeline

A modular pipeline for processing various data sources into standardized BabyLM multilingual datasets.

## Overview

This pipeline provides a flexible framework for:
1. Processing text data from any source into BabyLM format
2. Applying various preprocessing strategies (including LLM-based filtering)
3. Creating language-specific datasets where each document has its own metadata
4. Uploading to HuggingFace Hub


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

Each dataset contains documents with the following fields:

| Column | Description | Example Values |
|--------|-------------|----------------|
| text | Document text (preserves capitalization and paragraphs) | "This is a story.\n\nIt has multiple paragraphs." |
| category | Content type for this document | child-directed-speech, educational, subtitles, etc. |
| data-source | Original source of this document | OpenSubtitles, CHILDES, etc. |
| script | Writing system | latin, cyrillic, arabic, etc. |
| age-estimate | Target age for this document | "4", "12-17", "n/a" |
| license | License for this document | cc-by, cc-by-sa, etc. |
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

### With Document-Specific Metadata

Create a metadata JSON file that maps document IDs to specific metadata:

```json
{
  "doc1": {
    "category": "child-books",
    "age_estimate": "4-6",
    "source_url": "https://example.com/book1"
  },
  "doc2": {
    "category": "educational",
    "age_estimate": "8-10",
    "license": "cc-by-sa"
  }
}
```

Then use it:

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MixedSources" \
    --category "educational" \
    --texts-dir "./texts" \
    --script latin \
    --age-estimate "6-12" \
    --license "cc-by" \
    --metadata-file "./document_metadata.json"
```

### Preprocessing Options

The pipeline now **preserves capitalization and paragraph structure by default**. 

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
    --fix-unicode
```

#### If You Need Lowercasing
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
    --lowercase  # Explicitly enable lowercasing
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

The OpenSubtitles processor:
- Preserves capitalization by default
- Maintains document structure where possible
- Removes timestamps and stage directions
- Each subtitle file becomes a document with its own metadata

## Preprocessor Types

- **text**: Basic text preprocessing
- **subtitle**: Specialized for subtitle files (removes timestamps, stage directions)
- **transcript**: For dialogue transcripts (removes speaker labels, annotations)
- **llm**: Uses language models for quality filtering

## Text Structure Preservation

The pipeline preserves important text structure:

1. **Capitalization**: Maintained by default (use `--lowercase` to change)
2. **Paragraphs**: Double newlines (`\n\n`) indicate paragraph breaks
3. **Sentences**: Single newlines (`\n`) separate sentences within paragraphs

This preservation is important for:
- Proper nouns and sentence beginnings
- Document structure and context switches
- Natural reading flow

## Categories

Valid categories for documents:
- `child-directed-speech`: Direct speech to children
- `educational`: Educational content for children
- `child-books`: Children's literature
- `child-wiki`: Child-friendly encyclopedic content
- `child-news`: News adapted for children
- `subtitles`: TV/movie subtitles
- `qed`: QED educational content
- `child-available-speech`: Other dialogue/speech accessible to children

## Output Structure

Each processed dataset creates:
```
babylm-{language}/
├── texts/                    # Individual text files
├── dataset_metadata.json     # Complete metadata
├── babylm-{language}_dataset.csv     # Final dataset (CSV)
├── babylm-{language}_dataset.parquet # Final dataset (Parquet)
└── README.md                # Dataset card
```

## License


The pipeline code is provided as-is. Individual documents in datasets have their own licenses as specified in the metadata.

