---
task_categories:
- text-generation
language:
- {language}
license: {license_metadata}
size_categories:
- {size_category}
dataset_info:
  features:
    - name: text
      dtype: string
    - name: doc-id
      dtype: string
    - name: category
      dtype: string
    - name: data-source
      dtype: string
    - name: script
      dtype: string
    - name: age-estimate
      dtype: string
    - name: license
      dtype: string
    - name: misc
      dtype: string
    - name: num-tokens
      dtype: int64
    - name: language
      dtype: string
---

# {dataset_name}

## Dataset Description

This dataset is part of the BabyLM multilingual collection.   
More information at: [babylm.github.io/multilingual](https://babylm.github.io/multilingual)

### Dataset Summary

- **Language:** {language}
- **Script:** {script_display}
- **Tier:** {dataset_tier}
- **Byte Premium Factor:** {byte_premium_factor:.6f}
- **Size (MB):** {dataset_size:.2f}
- **Expected Size (MB):** {expected_size:.2f}
- **Number of Documents:** {num_documents:,}
- **Total Tokens:** {total_tokens:,}
- **Tokenizer:** {tokenizer_name}

### Tokens Per Category

{tokens_per_category_content}

### Data Fields

- `text`: The document text
- `doc-id`: Unique identifier for the document
- `category`: Type of content (e.g., child-directed-speech, educational, etc.)
- `data-source`: Original source of the data
- `script`: Writing system used (ISO 15924)
- `age-estimate`: Target age or age range
- `license`: Data license
- `misc`: Additional metadata (JSON string)
- `num-tokens`: Number of tokens per item (based on white-space split)
- `language`: Language code (ISO 639-3)

### Licensing Information

Please see license in individual documents

### Data Sources & Attribution

{data_sources_readme}

### Data Curators

{contributors_readme}