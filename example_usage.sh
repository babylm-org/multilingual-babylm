#!/bin/bash

# Example 1: Basic usage - process any text files
python main_pipeline.py \
    --language eng \
    --data-source "MyCustomTexts" \
    --category "educational" \
    --texts-dir "./my_text_files" \
    --script Latn \
    --age-estimate "6-12" \
    --license "cc-by" \
    --upload \
    --repo-id "username/babylm-eng"

# Example 2: Mixed-source dataset with document-specific metadata
# First create a metadata file that specifies different properties per document
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

python main_pipeline.py \
    --language eng \
    --data-source "MixedSources" \
    --category "educational" \
    --texts-dir "./mixed_texts_example" \
    --script Latn \
    --age-estimate "4-10" \
    --license "cc-by" \
    --metadata-file "./mixed_metadata.json" \
    --preprocess \
    --preprocessor-type text \
    --fix-unicode \
    --upload \
    --repo-id "username/babylm-eng"

# Example 3: Process with basic preprocessing (preserves caps and paragraphs by default)
python main_pipeline.py \
    --language fra \
    --data-source "FrenchStories" \
    --category "child-books" \
    --texts-dir "./french_stories" \
    --script Latn \
    --age-estimate "4-8" \
    --license "cc-by-sa" \
    --preprocess \
    --preprocessor-type text \
    --fix-unicode

# Example 4: Process with lowercasing (when explicitly needed)
python main_pipeline.py \
    --language fra \
    --data-source "FrenchPoems" \
    --category "child-books" \
    --texts-dir "./french_poems" \
    --script Latn \
    --age-estimate "6-10" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type text \
    --lowercase \
    --fix-unicode

# Example 5: Process subtitles with subtitle-specific preprocessing
python main_pipeline.py \
    --language deu \
    --data-source "GermanSubtitles" \
    --category "subtitles" \
    --texts-dir "./german_subs" \
    --script Latn \
    --age-estimate "n/a" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type subtitle \
    --remove-timestamps \
    --remove-stage-directions

# Example 6: Process CHILDES transcripts
python main_pipeline.py \
    --language nld \
    --data-source "CHILDES-Dutch" \
    --category "child-directed-speech" \
    --texts-dir "./childes_dutch" \
    --script Latn \
    --age-estimate "2-5" \
    --license "cc-by-sa" \
    --metadata-file "./childes_metadata.json" \
    --preprocess \
    --preprocessor-type transcript

# Example 7: Use LLM filtering for quality control
python main_pipeline.py \
    --language spa \
    --data-source "SpanishWebTexts" \
    --category "child-available-speech" \
    --texts-dir "./spanish_web_texts" \
    --script Latn \
    --age-estimate "8-14" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type llm \
    --llm-model "llama3.2" \
    --llm-prompt "Is this text appropriate and educational for Spanish-learning children aged 8-14? Consider vocabulary level, content appropriateness, and educational value. Respond with JSON: {\"score\": 0-1, \"reason\": \"explanation\"}" \
    --llm-filter-threshold 0.8

# Example 8: Process OpenSubtitles using the dedicated script
python process_opensubtitles.py \
    --language af \
    --script Latn \
    --batch-size 100 \
    --upload \
    --repo-id "username/babylm-afr"

# Example 9: Process OpenSubtitles with lowercasing (non-default)
python process_opensubtitles.py \
    --language ita \
    --script Latn \
    --lowercase \
    --batch-size 50 \
    --upload \
    --repo-id "username/babylm-ita"

# Example 10: Chain preprocessing - first extract, then additional processing
# Step 1: Extract texts without preprocessing
python process_opensubtitles.py \
    --language por \
    --script Latn \
    --no-preprocess \
    --keep-zip

# Step 2: Apply custom preprocessing with LLM filtering
python main_pipeline.py \
    --language por \
    --data-source "OpenSubtitles" \
    --category "subtitles" \
    --texts-dir "./output/por/preprocessed_texts" \
    --script Latn \
    --age-estimate "n/a" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type llm \
    --llm-prompt "Clean this Portuguese subtitle text for language learning. Remove any profanity or inappropriate content. Preserve capitalization and paragraph structure. Return JSON with score and cleaned text." \
    --upload \
    --repo-id "username/babylm-por-filtered"

# Example 11: Create a multi-age educational dataset
# Create metadata for different age groups
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

python main_pipeline.py \
    --language eng \
    --data-source "K12Education" \
    --category "educational" \
    --texts-dir "./k12_texts" \
    --script Latn \
    --age-estimate "4-13" \
    --license "cc-by-sa" \
    --metadata-file "./edu_metadata.json" \
    --source-url "https://example-education.org" \
    --misc '{"curriculum": "common-core", "year": "2024"}' \
    --upload \
    --repo-id "username/babylm-eng"

# Example 12: Process data with custom preprocessing function
# This shows how to add custom preprocessing while preserving structure
python main_pipeline.py \
    --language zho \
    --data-source "ChineseChildrenStories" \
    --category "child-books" \
    --texts-dir "./chinese_stories" \
    --script "Hans" \  # ISO 15924 code for "Simplified Chinese"
    --age-estimate "5-10" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type text \
    --fix-unicode \
    --no-lowercase  # Explicitly preserve case (though it's default)