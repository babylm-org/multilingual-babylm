# BabyLM Multilingual Dataset Pipeline

A modular pipeline for processing various data sources into standardized BabyLM-like multilingual datasets.

## Overview

This pipeline provides a flexible framework for:

1. Processing text data from any source into the BabyBabelLM format
2. Applying robust, category-specific preprocessing strategies
3. Creating language-specific datasets where each document has its own metadata
4. Uploading to HuggingFace Hub with included pull-request functionality


## Contributing

### Data Contributions

BabyBabelLM is a living resource, created, maintained, and updated by language communities.   
We encourage **community contributions**, in two different forms: 
1. Updating the dataset for one of the languages in BabyBabelLM, by adding, removing, or correcting data
2. Adding data for a new language not present in BabyBabelLM

For the first option, there is built-in functionality in the pipeline that will automatically create a pull-request on the HuggingFace Hub. Afterwards, a member of BabyBabelLM will review and incorporate changes into the dataset. To use the PR functionality in the pipeline see **Usage** below.

For the second option, please contact us via opening an issue, so we can first create the corresponding BabyBabelLM dataset for the language, which can then be updated by pull request like above.


### Code Contributions

We also welcome contributions to the pipeline code in the form of Github pull requests.



## Project Structure

```
├── resources
│   ├── byte_coefs_20240233.tsv     # Byte-premium factors
│   ├── contributors.yaml           # Information about data contributors
│   ├── data_sources.yaml           # Detailed data-source information for each dataset repo
│   ├── readme_template.txt         # Template to generate readme for each repo
│   └── repo_comments.yaml          # Additional comments for each language repo
├── scripts
│   ├── examples.sh                 # Some examples for using the pipeline
│   └── process_opensubtitles.sh    # Process OpenSubtitles data
└── src
    ├── babylm_dataset_builder.py   # BabyLM dataset builder
    ├── hf_uploader.py              # HuggingFace upload utilities
    ├── language_filter.py          # Language and script filtering (GlotLID)
    ├── language_scripts.py         # ISO 15924 script code mapping utilities
    ├── loader.py                   # Data loading utilities
    ├── logging_utils.py            # Logging utilities
    ├── multilingual_res/           # Multilingual resources utilities
    │   ...
    ├── opensubtitles/              # OpenSubtitles data processing
    │   ...
    ├── pad_dataset.py              # Pad dataset with various data
    ├── pad_language_specific.py    # Pad with language-specific resources
    ├── pad_utils.py                # Padding utilities
    ├── pipeline.py                 # Main pipeline code
    ├── preprocessor.py             # Unified text preprocessing utilities
    ├── preprocessor_utils.py       # Standalone preprocessing functions
    └── update_dataset.py           # Easy dataset update utility code
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

## Usage
Below we include some scripts for basic usage of the pipeline. 
- Some more usage examples can also be found at `resources/usage_examples`.
- For a template script of updating a dataset using the pipeline see also: `src/update_dataset.py`.


```sh
python src/pipeline.py \
  --language eng \
  --script Latn \
  --data-path examples/dataset.json \
  --data-type json \
  --preprocess \
  --pad \
  --repo-id username/babylm-eng \
  --upload
```

```python
from src.pipeline import process_dataset

process_dataset(
  language="eng", 
  script="Latn",
  data_path="examples/dataset.json"
  preprocess=True,
  pad_opensubtitles=True
  repo_id="username/babylm-eng"
  upload=True
)
```

After processing, the output HuggingFace-compatible dataset will be created in a new directory named `babylm-{language}` at a parent directory `babylm_datasets/`. Each dataset has the following structure:

```
babylm-{language}/
├── dataset_metadata.json               # Complete metadata
├── babylm-{language}_dataset.csv       # Final dataset (CSV)
├── babylm-{language}_dataset.parquet   # Final dataset (Parquet)
├── log.txt                             # Logging messages
└── README.md                           # Dataset card
```

For more advanced options, below we give detailed descriptions of each command-line argument in the pipeline. 


### Mandatory Arguments 
- `--language, -l`: ISO 639-3 language code
- `--script`: Must be a valid ISO 15924 code (e.g., Latn, Cyrl, Arab, etc.)
- `--data-path, -t`: Path to data directory or file
- `--data-type`: Loader type for input data

  > Choices for `data-type` are:
  >  - `text`: Directory of `.txt` files (each file is a document)
  >  - `csv`: CSV file (each row is a document)
  >  - `json`: JSON file (list of dicts, each dict is a document)
  >  - `jsonl`: JSON Lines file (each line is a JSON dict)
  >  - `hf`: HuggingFace dataset (specify path to dataset or dataset ID in `--data-path`)

### Basic Functionality
- `--upload`: Upload to HuggingFace after processing
- `--repo-id`: HuggingFace repo ID (e.g., `username/babylm-eng`)
- `--preprocess, --preprocess-text`: Enable text preprocessing
- `--remove-previous-padding`: Remove previously added padding for the given language
- `--pad, --pad-opensubtitles`: Enable padding with OpenSubtitles, FineWeb-C, Wikipedia, or language-specific resources

  > For padding, we use primarly subtitle data from the coressponding HF repo: `BabyLM-community/babylm-{lang_code}-subtitles`, where lang_code is the ISO 639-1 code for the language specified in `--language`. If subtitle data is not available or runs out, we use data from Wikipedia and FineWeb-C. Finally, some language-specific language resources can be defined and used in `src/pad_language_specific.py`.
  > 
  > To calculate the required padding dataset size for a 100M words of English equivalent dataset, we use Byte Premium factors [(paper link)](https://aclanthology.org/2024.sigul-1.1.pdf).


### Creating Pull Requests
-  `--create-pr`: Create a pull-request (PR) on the HF repo for this dataset
-  `--pr-title` : Title of the PR
-  `--pr-description`: Description of the PR

    >The  pull requrest will be created in the repo supplied in the `--upload` argument. 


### Advanced Functionality 
- `--enable-script-update`: Enable script identification and updates (default: disabled)
- `--script-update-all`: Update scripts for the whole dataset (default only new documents)
- `--byte-premium-factor`: Byte-premium factor override (float). Byte-premium factors are automatically pulled from the resources directory. To override, or in case of a missing factor, you can use this argument. 
- `--tokenizer-name`: Tokenizer name for token counting (for languages like Chinese, Japanese, Korean)
- `--merge`: Merge existing dataset instead of overwriting
- `--logfile`: Logging filepath
- `--enable-language-filtering`: Enable language and script filtering using GlotLID v3
- `--language-filter-threshold`: Minimum confidence threshold for language filtering (0.0–1.0, default 0.8)

  > For language/script filtering we use GlotLID v3. The the minimum confidence threshold can be set with `--language-filter-threshold` (default: 0.8).
  > During filtering, matching files will be used for dataset creation while mismatched files are excluded from the final dataset.


  > Sometimes you might wish to incrementally add data to a repo with multiple passes. This is necessary when data might have from multiple resources, e.g., a HuggingFace repo, a .json file, some text files etc. To achieve this, use the `--merge` option without including `--upload`, and the data added from multiple passes will be merged in `babylm_datasets/babylm-{language}`. The data then be uploaded with a final use of `--upload`.

  > Without using `--merge` each time you run the pipeline data in `babylm_datasets/babylm-{language}` is overwritten. 


### Multilingual Resources
You can automatically fetch and add public multilingual resources to your dataset using the following options:
These options can be combined with any other data source. The fetched documents will be merged and deduplicated with your own data.
- `--add-ririro-data`: Add Ririro resource for the given language
- `--add-glotstorybook-data`: Add GlotStoryBook resource for the given language
- `--add-childwiki-data`: Add ChildWiki resource for the given language
- `--add-childes-data`: Add Childes resource for the given language
- `--remove-previous-ririro-data`: Remove previously added Ririro resource
- `--remove-previous-glotstorybook-data`: Remove previously added GlotStoryBook resource
- `--remove-previous-childwiki-data`: Remove previously added ChildWiki resource
- `--remove-previous-childes-data`: Remove previously added Childes resource

### Supply Document Metadata

- `--data-source, -s`: Data source name (e.g., OpenSubtitles, CHILDES, etc.)
- `--category, -c`: Content category 
- `--age-estimate`: Age estimate (e.g., `4`, `12-17`, `n/a`)
- `--license`: License (e.g., cc-by, cc-by-sa)
- `--misc`: Additional metadata as JSON string
- `--metadata-file`: JSON file with document metadata

  > The arguments `--data-source`, `--age-estimate`, `--license`, `--category` and `--misc` are optional. If provided, they supplement the document metadata by filling    missing values. They never override document-specific metadata.

  > For each document, all metadata fields are automatically extracted from the file (for CSV, JSON, JSONL, or HF datasets)
  > You can provide a separate metadata file in JSON format using `--metadata-file` to add or override metadata for specific documents. The JSON should map document IDs to metadata fields. If a document's ID is not found in the metadata file, the pipeline will use the values provided via command-line arguments or extracted from the file.



## License

The pipelien code is released under the MIT license. See `LICENSE` file.
Individual documents in datasets have their own licenses as specified in the metadata.

## Appendix

### Dataset Format

Each dataset contains documents with the following fields:

| Column       | Description                                                      | Example Values                                    |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------- |
| doc-id       | Unique document identifier                                       | SHA256 hash string                                |
| text         | Document text (preserves capitalization and paragraphs)          | "This is a story.\n\nIt has multiple paragraphs." |
| category     | Content type for this document                                   | See below                                         |
| data-source  | Original source of this document                                 | OpenSubtitles, CHILDES, etc. (default: "unknown") |
| script       | Writing system (ISO 15924 code)                                  | `Latn`, `Cyrl`, etc.                              |
| language     | Language of the text (ISO 693-3 code)                            | `ell`, `bul`, etc.                                |
| age-estimate | Target age for this document                                     | "4", "12-17", (default: "n/a")                    |
| license      | License for this document                                        | cc-by, cc-by-sa, etc.                             |
| num-tokens   | Number of tokens in the text (white-space or tokenizer-based)    | 2500, 1154, etc.                                  |
| misc         | Additional metadata                                              | JSON string with extra info                       |

### Categories
| Category                | Description                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------- | 
| `child-directed-speech` | Speech directed to children and speech produced by children                                     |
| `child-available-speech`| Speech children are exposed to without being the target recipients (e.g., adult conversations)  |
| `educational`           | School textbooks, exams, and other educational material designed for children                   |
| `child-books`           | Books and stories created for children                                                          |
| `child-wiki`            | Children wiki articles                                                                          |
| `child-news`            | News directed to children                                                                       | 
| `subtitles`             | Subtitles for child-appropriate material (e.g., children TV shows).                             |
| `qed`                   | Subtitles from the QED dataset.                                                                 |
| `padding-wikipedia`     | Wikipedia articles                                                                              |
| `padding-[placeholder]` | Other forms of padding, used primarily for low-resource languages.                              |

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



