#!/bin/bash

# Example 1: Basic usage - process any text files
time python pipeline.py \
    --language eng \
    --category "educational" \
    --loader-path "./my_text_files" \
    --loader-type text \
    --script Latn \
    --license "cc-by"

# Example 1b: Basic usage with preprocessing
time python pipeline.py \
    --language eng \
    --category "educational" \
    --loader-path "./my_text_files" \
    --loader-type text \
    --script Latn \
    --license "cc-by" \
    --preprocess-text

# Example 2: Mixed-source dataset with document-specific metadata
cat > mixed_metadata.json << EOF
{
  "story1": {
    "category": "child-books",
    "age_estimate": "4-6",
    "data_source": "PublicDomainStories",
    "source_url": "https://example.com/story1"
  },
  "lesson1": {
    "category": "educational", 
    "age_estimate": "8-10",
    "data_source": "OpenEducation",
    "license": "cc-by-sa"
  },
  "transcript1": {
    "category": "child-directed-speech",
    "age_estimate": "2-4",
    "data_source": "CHILDES"
  }
}
EOF

time python pipeline.py \
    --language eng \
    --category "educational" \
    --loader-path "./mixed_texts_example" \
    --loader-type text \
    --script Latn \
    --license "cc-by" \
    --metadata-file "./mixed_metadata.json"

# Example 2b: Mixed-source dataset with document-specific metadata and preprocessing
time python pipeline.py \
    --language eng \
    --category "educational" \
    --loader-path "./mixed_texts_example" \
    --loader-type text \
    --script Latn \
    --license "cc-by" \
    --metadata-file "./mixed_metadata.json" \
    --preprocess-text

# Example 3: Process with language filtering
time python pipeline.py \
    --language fra \
    --category "child-books" \
    --loader-path "./french_stories" \
    --loader-type text \
    --script Latn \
    --license "cc-by-sa" \
    --enable-language-filtering \
    --language-filter-threshold 0.85

# Example 4: Process subtitles
time python pipeline.py \
    --language deu \
    --category "subtitles" \
    --loader-path "./german_subs" \
    --loader-type text \
    --script Latn \
    --license "cc-by"

# Example 5: Process CHILDES transcripts
time python pipeline.py \
    --language nld \
    --category "child-directed-speech" \
    --loader-path "./childes_dutch" \
    --loader-type text \
    --script Latn \
    --license "cc-by-sa" \
    --metadata-file "./childes_metadata.json"

# Example 6: Create a multi-age educational dataset
cat > edu_metadata.json << EOF
{
  "kindergarten_lesson1": {"age_estimate": "4-5", "category": "educational"},
  "kindergarten_lesson2": {"age_estimate": "4-5", "category": "educational"},
  "elementary_lesson1": {"age_estimate": "6-8", "category": "educational"},
  "elementary_lesson2": {"age_estimate": "6-8", "category": "educational"},
  "middle_school_lesson1": {"age_estimate": "11-13", "category": "educational"},
  "middle_school_lesson2": {"age_estimate": "11-13", "category": "educational"}
}
EOF

time python pipeline.py \
    --language eng \
    --category "educational" \
    --loader-path "./k12_texts" \
    --loader-type text \
    --script Latn \
    --license "cc-by-sa" \
    --metadata-file "./edu_metadata.json"

# Example 7: Process data with language filtering and upload
time python pipeline.py \
    --language ind \
    --category child-news \
    --loader-path ./articles_cleaned_txt \
    --loader-type text \
    --script Latn \
    --license cc-by \
    --enable-language-filtering \
    --language-filter-threshold 0.8 \
    --upload \
    --repo-id "username/babylm-ind"