# BabyLM Multilingual Dataset Pipeline

A modular pipeline for processing various data sources into standardized BabyLM multilingual datasets.

## Overview

This pipeline provides a flexible framework for:

1. Processing text data from any source into BabyLM format
2. Applying robust, category-specific preprocessing strategies
3. Creating language-specific datasets where each document has its own metadata
4. Uploading to HuggingFace Hub

## Project Structure

```
├── pipeline.py                  # Main generic pipeline
├── babylm_dataset_builder.py    # BabyLM dataset builder
├── preprocessor.py              # Unified text preprocessing utilities
├── preprocessor_utils.py        # Standalone preprocessing functions
├── loader.py                    # Data loading utilities
├── hf_uploader.py               # HuggingFace upload utilities
├── language_filter.py           # Language and script filtering (GlotLID)
├── language_scripts.py          # ISO 15924 script code mapping utilities
├── pad_dataset.py               # Padding with OpenSubtitles data
├── multilingual_res/            # Fetch and integrate public multilingual resources
├── requirements.txt             # Python dependencies
├── upload_basic_json.sh         # Basic usage with a .json dataset file
└── example_usage.sh             # Usage examples

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

| Column       | Description                                                      | Example Values                                    |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------- |
| doc_id       | Unique document identifier                                       | SHA256 hash string                                |
| text         | Document text (preserves capitalization and paragraphs)          | "This is a story.\n\nIt has multiple paragraphs." |
| category     | Content type for this document                                   | See below                                         |
| data-source  | Original source of this document                                 | OpenSubtitles, CHILDES, etc. (default: "unknown") |
| script       | Writing system (ISO 15924 code as input, formal name in dataset) | `Latn`:Latin, `Cyrl`:Cyrillic, etc.               |
| age-estimate | Target age for this document                                     | "4", "12-17", (default: "n/a")                    |
| license      | License for this document                                        | cc-by, cc-by-sa, etc.                             |
| misc         | Additional metadata                                              | JSON string with extra info                       |

### Categories

- `child-directed-speech`
- `educational`
- `child-books`
- `child-wiki`
- `child-news`
- `subtitles`
- `qed`
- `child-available-speech`
- `simplified-text`
- `padding`

## Usage

### Basic Usage (Any Text Source)

- Use `--data-path` to specify the path to your data (directory or file).
- Use `--data-type` to specify the input format: `text`, `csv`, `json`, `jsonl`, or `hf` (HuggingFace dataset).
- `--data-source`, `--age-estimate`, `--license`, `--category` and `--misc` are optional. If provided, they supplement the document metadata by filling missing values. They never override document-specific metadata.
- `--script` (in ISO 15924) and `--language` (in ISO 693-3) should always be provided.
- After processing, the output HuggingFace-compatible dataset will be created in a new directory named `babylm-{language}` inside a parent folder called `babylm_datasets` (e.g., `babylm_datasets/babylm-eng/`).
- **By default, new data will be merged with any existing dataset. To overwrite, use `--overwrite`.**

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --category "educational" \
    --data-path "./my_text_files" \
    --data-type text \
    --license "cc-by"
```

#### Example: With Document-Specific Metadata

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --category "educational" \
    --data-path "./texts" \
    --data-type text \
    --license "cc-by" \
    --metadata-file "./document_metadata.json"
```

#### Example: Upload to HuggingFace

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --category "educational" \
    --data-path "./texts" \
    --data-type text \
    --license "cc-by" \
    --upload \
    --repo-id "username/babylm-eng"
```

#### Example: With Preprocessing Enabled

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --category "educational" \
    --data-path "./my_text_files" \
    --data-type text \
    --license "cc-by" \
    --preprocess-text
```

#### Example: With Document-Specific Metadata and Preprocessing

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --category "educational" \
    --data-path "./texts" \
    --data-type text \
    --license "cc-by" \
    --metadata-file "./document_metadata.json" \
    --preprocess-text
```

### Multilingual Resource Integration

You can automatically fetch and add public multilingual resources to your dataset using the following options:

- `--add-ririro-data`: Fetch and add Ririro children's books for the specified language.
- `--add-glotstorybook-data`: Fetch and add GlotStoryBook storybooks for the specified language.
- `--add-childwiki-data`: Fetch and add ChildWiki child-friendly wiki content for the specified language.
- `--add-childes-data`: Fetch and add child-directed speech transcripts from the CHILDES database for the specified language.

These options can be combined with any other data source. The fetched documents will be merged and deduplicated with your own data.

**Example: Combine multilingual resources with your own data:**

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --data-path ./my_text_files \
    --data-type text \
    --add-ririro-data \
    --add-glotstorybook-data \
    --add-childwiki-data
```

**Example: Use only multilingual resources (no data-path or data-type):**

```bash
python pipeline.py \
    --language kor \
    --script Kore \
    --add-ririro-data \
    --add-glotstorybook-data \
    --add-childwiki-data \
    --add-childes-data
```

### Dataset Overwrite and Merging Behavior

- **Default:** If a dataset with the same language already exists in the output directory, new data will be merged with the existing dataset. Duplicate documents (by content) will be automatically deduplicated by `doc_id`.
- **Overwrite:** To overwrite and replace the existing dataset entirely (instead of merging), pass the `--overwrite` flag to the pipeline:

```bash
python pipeline.py \
    --language eng \
    --script Latn \
    --category "educational" \
    --data-path "./my_text_files" \
    --data-type text \
    --license "cc-by" \
    --overwrite
```

- **Deduplication:** The pipeline automatically removes exact duplicate documents based on the final (preprocessed) text content. Deduplication is performed:
  - Before saving the dataset, after all preprocessing, filtering, and optional padding steps.
  - When merging with existing datasets (e.g., on repeated runs or uploads), deduplication is also applied to the combined data to ensure no duplicate documents remain, even across multiple runs.
  - Deduplication uses a SHA256 hash of the final text, so only truly identical documents are removed.

### Loader Types

- `text`: Directory of `.txt` files (each file is a document)
- `csv`: CSV file (each row is a document)
- `json`: JSON file (list of dicts, each dict is a document)
- `jsonl`: JSON Lines file (each line is a JSON dict)
- `hf`: HuggingFace dataset (specify path to dataset or dataset ID in `--data-path`)

### Preprocessing Steps

Preprocessing is optional and only applied if `--preprocess-text` is passed. When enabled, preprocessing is category-specific. The steps are:

#### General Preprocessing (applied to all documents):

- Unicode normalization (using `ftfy`)
- Whitespace normalization (preserve paragraphs, collapse multiple spaces)

#### Category-Specific Preprocessing:

- **TranscriptPreprocessor** (`child-directed-speech`, `child-available-speech`):
  - Remove annotation lines starting with `%Speaker:` (any case)
  - Normalize punctuation
- **SubtitlePreprocessor** (`subtitles`):
  - Remove speaker labels at the start of lines (e.g., `John:`, `MOTHER:`, `John Smith:`)
  - Remove music note symbols (♪, ♫)
  - Remove stage directions (e.g., `[Music]`)
  - Remove timestamps (e.g., `[00:00:00]`)
  - Normalize punctuation
- **BookPreprocessor** (`educational`, `child-books`, `child-wiki`, `child-news`, `simplified-text`, `padding`):
  - Remove XML/HTML tags
  - Normalize punctuation
  - Remove URLs
- **QEDPreprocessor** (`qed`):
  - Remove XML/HTML tags
  - Normalize punctuation
  - Remove URLs

See `preprocessor.py` and `preprocessor_utils.py` for details.

### Document Metadata

- For each document, all metadata fields are automatically extracted from the file (for CSV, JSON, JSONL, or HF datasets) or from the filename (for text files, using the filename as `doc_id`).
- You can provide a separate metadata file in JSON format using `--metadata-file` to add or override metadata for specific documents. The JSON should map document IDs to metadata fields. If a document's ID is not found in the metadata file, the pipeline will use the values provided via command-line arguments or extracted from the file.

### Language and Script Filtering

- Enable language/script filtering using GlotLID v3 with `--enable-language-filtering`.
- Set the minimum confidence threshold with `--language-filter-threshold` (default: 0.8).

#### Example:

```bash
python pipeline.py \
    --language ind \
    --script Latn \
    --category child-news \
    --data-path ./articles_cleaned_txt \
    --data-type text \
    --license cc-by \
    --enable-language-filtering \
    --language-filter-threshold 0.8
```

- Matching files will be used for dataset creation.
- Mismatched files are excluded from the final dataset.

### Pad with OpenSubtitles data

```bash
python pipeline.py \
    --language ind \
    --script Latn \
    --category child-news \
    --data-path ./articles_cleaned_txt \
    --data-type text \
    --license cc-by \
    --pad-opensubtitles
```

- Uses subtitle data from the coressponding HF repo: `BabyLM-community/babylm-{lang_code}-subtitles`, where lang_code is the ISO 639-1 code for the language specified in `--language`.
- Use Byte Premium factorsto calculate the required padding dataset size for a 100M words of English equivalent dataset [(paper link)](https://aclanthology.org/2024.sigul-1.1.pdf).

## Output Structure

Each processed dataset creates:

```
babylm-{language}/
├── dataset_metadata.json     # Complete metadata
├── babylm-{language}_dataset.csv     # Final dataset (CSV)
├── babylm-{language}_dataset.parquet # Final dataset (Parquet)
└── README.md                # Dataset card
```

## License

The pipeline code is provided as-is. Individual documents in datasets have their own licenses as specified in the metadata.
