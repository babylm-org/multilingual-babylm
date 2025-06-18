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
├── hf_uploader.py               # HuggingFace upload utilities
├── process_opensubtitles.py     # OpenSubtitles-specific wrapper
├── opensubtitles_processor.py   # OpenSubtitles processing module
├── language_filter.py           # Language and script filtering (GlotLID)
├── language_scripts.py          # ISO 15924 script code mapping utilities
├── requirements.txt             # Python dependencies
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
| text         | Document text (preserves capitalization and paragraphs)          | "This is a story.\n\nIt has multiple paragraphs." |
| category     | Content type for this document                                   | See below                                         |
| data-source  | Original source of this document                                 | OpenSubtitles, CHILDES, etc.                      |
| script       | Writing system (ISO 15924 code as input, formal name in dataset) | `Latn`:Latin, `Cyrl`:Cyrillic, etc.               |
| age-estimate | Target age for this document                                     | "4", "12-17", "n/a"                               |
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

## Usage

### Basic Usage (Any Text Source)

- If you do **not** use `--preprocess`, the directory specified by `--texts-dir` must contain plain text files with the `.txt` extension. Each `.txt` file will be treated as a separate document.
  - After processing, the output HuggingFace-compatible dataset will be created in a new directory named `babylm-{language}` inside a parent folder called `babylm_datasets` (e.g., `babylm_datasets/babylm-eng/`).

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MyDataSource" \
    --category "educational" \
    --texts-dir "./path/to/texts" \
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by"
```

- If you use `--preprocess` with `--preprocessor-type text`, the pipeline will preprocess all `.txt` files in the directory specified by `--texts-dir` and build the dataset from the preprocessed files. The original files are never overwritten; preprocessed files are written to a new directory named `preprocessed_{data-source}_{language}` (e.g., `preprocessed_MyDataSource_eng`).

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MyDataSource" \
    --category "educational" \
    --texts-dir "./path/to/texts" \
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type text
```

- If you use `--preprocess` with `--preprocessor-type csv`, the pipeline will extract text and metadata from the CSV file specified by `--texts-dir`, write each row's text to a `.txt` file in a new preprocessed directory, and build the dataset from those files. Only fields relevant to the BabyLM dataset (category, data_source, script, age_estimate, license, misc, source_url, source_identifier) are extracted as metadata.
- If you use `--preprocess` with `--preprocessor-type hf`, the pipeline will extract text and metadata from the HuggingFace dataset specified by `--texts-dir` (dataset ID), write each example's text to a `.txt` file in a new preprocessed directory, and build the dataset from those files. Only fields relevant to the BabyLM dataset are extracted as metadata. You can specify the split with `--hf-dataset-split` (default: use the default split).
- If you do **not** use `--preprocess`, you cannot use CSV or HuggingFace datasets as input; only a directory of `.txt` files is supported.

#### Example: Using a CSV as Input

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MyCSVSource" \
    --category "educational" \
    --texts-dir "./mydata.csv" \
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type csv \
    --text-field "text"
```

#### Example: Using a HuggingFace Dataset as Input

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MyHFSource" \
    --category "educational" \
    --texts-dir "my_hf_dataset_id" \
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type hf \
    --text-field "text" \
    --hf-dataset-split "train"
```

- The pipeline will extract the text and relevant metadata fields from the dataset, write each example to a `.txt` file, and build the BabyLM dataset from those files.
- If you do not specify `--hf-dataset-split`, no split will be used.

### With Document-Specific Metadata

- You can provide a JSON file mapping document IDs (filenames without `.txt`) to specific metadata fields.
- If a document's ID is not found in the metadata file, the pipeline will use the values provided via command-line arguments for that document.
- If using CSV or HuggingFace dataset as Input, by default the metadata is extracted from the CSV file/HF dataset.

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
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by" \
    --metadata-file "./document_metadata.json"
```

### Text Preprocessing

- If you enable preprocessing (with `--preprocess`), the pipeline will write the preprocessed `.txt` files to a new directory named `preprocessed_{data-source}_{language}` (e.g., `preprocessed_MyDataSource_eng`).
- The dataset will then be built from these preprocessed files, not the originals.
- The original files are never overwritten.

#### Available Custom Preprocessing Steps

You can enable additional custom preprocessing steps using the following flags:

- `--remove-urls`: Remove URLs from the text.
- `--normalize-punctuation`: Normalize punctuation (e.g., convert curly quotes to straight quotes, unify dashes, etc.).
- `--remove-xml-tags`: Remove XML/HTML tags from the text.
- `--replace-newline-within-paragraph`: Replace single newlines with a space within paragraphs (default: off). This is useful if you want to treat each paragraph as a single line, even if the original text had line breaks within paragraphs.
- `--tokenizer-name` : Specify the tokenizer you would like to use to count tokens (default to whitespace split). Support HuggingFace tokenizers.

You can combine these with other preprocessing options. For example:

```bash
python main_pipeline.py \
    --language eng \
    --data-source "MyDataSource" \
    --category "educational" \
    --texts-dir "./texts" \
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by" \
    --preprocess \
    --remove-urls \
    --normalize-punctuation \
    --remove-xml-tags \
    --replace-newline-within-paragraph
```

#### Basic Text Preprocessing

```bash
python main_pipeline.py \
    --language fra \
    --data-source "FrenchTexts" \
    --category "child-books" \
    --texts-dir "./texts" \
    --script Latn \
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
    --script Latn \
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
    --script Latn \
    --batch-size 100 \
    --upload \
    --repo-id "bhargavns/babylm-afr"
```

The OpenSubtitles processor:

- Preserves capitalization by default
- Maintains document structure where possible
- Removes timestamps and stage directions
- Each subtitle file becomes a document with its own metadata

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

## Preprocessor Types

- **text**: Basic text preprocessing
- **subtitle**: Specialized for subtitle files (removes timestamps, stage directions)
- **transcript**: For dialogue transcripts (removes speaker labels, annotations)
- **llm**: Uses language models for quality filtering
- **csv**: For CSV files; extracts text and relevant metadata fields from each row, writes each as a .txt file, and builds the dataset from those files (requires --preprocess)
- **hf**: For HuggingFace datasets; extracts text and relevant metadata fields from each example, writes each as a .txt file, and builds the dataset from those files (requires --preprocess)

## Text Structure Preservation

The pipeline preserves important text structure:

1. **Capitalization**: Maintained by default (use `--lowercase` to change)
2. **Paragraphs**: Double newlines (`\n\n`) indicate paragraph breaks
3. **Sentences**: Single newlines (`\n`) are maintained within paragraph

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

## Language and Script Filtering

The pipeline supports automatic language and script filtering using GlotLID v3. This ensures that only documents matching the desired language and script are included in the final dataset.

### How It Works

- Each document is segmented into reasonable-length chunks.
- GlotLID v3 predicts the language and script for each segment.
- The document's language and script are determined by majority vote **weighted by word count** (not just segment count).
- Only documents matching the specified language and script (with sufficient confidence) are included in the main dataset. Others are saved separately for inspection.

### Enabling Language Filtering

Add the following arguments to your pipeline command:

- `--enable-language-filtering` — Enable language/script filtering
- `--language-filter-threshold 0.8` — (Optional) Set the minimum confidence threshold (default: 0.8)

#### Example:

```bash
python main_pipeline.py \
    --language ind \
    --data-source "Bobo" \
    --category child-news \
    --texts-dir ./articles_cleaned_txt \
    --script Latn \
    --age-estimate "6-12" \
    --license cc-by \
    --enable-language-filtering \
    --language-filter-threshold 0.8
```

- Matching files will be used for dataset creation.
- Mismatched files are saved in a `filtered/mismatched/` subdirectory inside the output folder for later review.

### Output Structure with Filtering

When language filtering is enabled, the output directory will include:

```
babylm-{language}/
├── filtered/
│   ├── {language}/           # Matching files
│   └── mismatched/           # Files not matching language/script
│       ├── {pred_lang}_{pred_script}/
│       └── ...
├── texts/                    # Final dataset files
├── dataset_metadata.json     # Complete metadata
├── babylm-{language}_dataset.csv
├── babylm-{language}_dataset.parquet
└── README.md
```

### Notes

- Filtering is based on **majority of words** in the document, not just the number of segments.
- You can adjust the confidence threshold with `--language-filter-threshold`.
- Filtering is available for any data source processed with `main_pipeline.py`.

## License

The pipeline code is provided as-is. Individual documents in datasets have their own licenses as specified in the metadata.
