#!/bin/bash

# Example 1: Process any text files with command-line global metadata
time python pipeline.py \
    --language eng \
    --script Latn \
    --data-path "./my_text_files" \
    --data-type text \
    --category "educational" \
    --license "cc-by"\
    --data-source "PublicDomainStories" \
    --age-estimate "4-6" \
    --misc '{"source_url": "https://example.com/my_text_files"}'
  
# Example 2: Process dataset from file with a format of: json, jsonl, csv, or HF dataset
# command-line metadata can still be used, but are overridden by document-specific metadata
time python pipeline.py \
    --language eng \
    --script Latn \
    --data-path "./my_dataset.json" \
    --data-type json \
    --category "educational" # will be overriden by category value in json file, if specified there


# Example 3: Add document-specific metadata from json file
cat > mixed_metadata.json << EOF
{
  "story1": {
    "category": "child-books",
    "age-estimate": "4-6",
    "data-source": "PublicDomainStories",
    "source_url": "https://example.com/story1"
  },
  "lesson1": {
    "category": "educational", 
    "age-estimate": "8-10",
    "data-source": "OpenEducation",
    "license": "cc-by-sa"
  },
  "transcript1": {
    "category": "child-directed-speech",
    "age-estimate": "2-4",
    "data-source": "CHILDES"
  }
}
EOF

# provide directory of text files, document-specific metadata is provided in the json metadata file
# command-line metadata can still be used, but are overridden by document-specific metadata
time python pipeline.py \
    --language eng \
    --script Latn \
    --data-path "./mixed_texts_example" \
    --data-type text \
    --metadata-file "./mixed_metadata.json"\
    --category "educational" # will be overriden by category value in json file, if specified there


# Example 4: Add pre-processing and language filtering
time python pipeline.py \
    --language eng \
    --script Latn \
    --data-path "./my_dataset.json" \
    --data-type json \
    --preprocess-text \
    --enable-language-filtering \
    --language-filter-threshold 0.85


# Example 5: Process, filter and upload a dataset
time python pipeline.py \
    --language ind \
    --script Latn \
    --data-path ./articles_cleaned.json \
    --data-type json \
    --preprocess-text \
    --enable-language-filtering \
    --language-filter-threshold 0.8 \
    --upload \
    --repo-id "username/babylm-ind"



# Example 6: Pad dataset with OpenSubtitles data
# taking into account byte premiums
time python pipeline.py \
    --language ind \
    --script Latn \
    --data-path ./articles_cleaned.json \
    --data-type json \
    --preprocess-text \
    --enable-language-filtering \
    --language-filter-threshold 0.8 \
    --upload \
    --repo-id "username/babylm-ind" \
    --pad-opensubtitles # add padding with opensubtitles data

# ========== Examples for specific datasets ==========

# Example 7: Process subtitles
time python pipeline.py \
    --language deu \
    --script Latn \
    --data-path "./german_subs" \
    --data-type text \
    --license "cc-by" \
    --category "subtitles" \
    --data-source "OpenSubtitles" \
    --age-estimate "13-15" \
    --misc '{"source_url": "https://example.com/german_subs"}'


# Example 8: Process CHILDES transcripts
time python pipeline.py \
    --language nld \
    --script Latn \
    --data-path "./childes_dutch" \
    --data-type text \
    --category "child-directed-speech" \
    --license "cc-by-sa" \
    --metadata-file "./childes_metadata.json"

# Example 9: Create a multi-age educational dataset
cat > edu_metadata.json << EOF
{
  "kindergarten_lesson1": {"age-estimate": "4-5", "category": "educational"},
  "kindergarten_lesson2": {"age-estimate": "4-5", "category": "educational"},
  "elementary_lesson1": {"age-estimate": "6-8", "category": "educational"},
  "elementary_lesson2": {"age-estimate": "6-8", "category": "educational"},
  "middle_school_lesson1": {"age-estimate": "11-13", "category": "educational"},
  "middle_school_lesson2": {"age-estimate": "11-13", "category": "educational"}
}
EOF

time python pipeline.py \
    --language eng \
    --script Latn \
    --data-path "./k12_texts" \
    --data-type text \
    --category "educational" \
    --license "cc-by-sa" \
    --metadata-file "./edu_metadata.json"

