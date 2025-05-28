#!/bin/bash

# Example 1: Process any text files (generic)
python main_pipeline.py \
    --language eng \
    --data-source "MyCustomTexts" \
    --category "educational" \
    --texts-dir "./my_text_files" \
    --script latin \
    --age-estimate "6-12" \
    --license "cc-by" \
    --upload \
    --repo-id "bhargavns/test"

# Example 2: Process with basic preprocessing
python main_pipeline.py \
    --language fra \
    --data-source "FrenchStories" \
    --category "child-books" \
    --texts-dir "./french_stories" \
    --script latin \
    --age-estimate "4-8" \
    --license "cc-by-sa" \
    --preprocess \
    --preprocessor-type text \
    --lowercase \
    --fix-unicode

# Example 3: Process subtitles with subtitle-specific preprocessing
python main_pipeline.py \
    --language deu \
    --data-source "GermanSubtitles" \
    --category "subtitles" \
    --texts-dir "./german_subs" \
    --script latin \
    --age-estimate "n/a" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type subtitle \
    --remove-timestamps \
    --remove-stage-directions

# Example 4: Process CHILDES transcripts
python main_pipeline.py \
    --language nld \
    --data-source "CHILDES-Dutch" \
    --category "child-directed-speech" \
    --texts-dir "./childes_dutch" \
    --script latin \
    --age-estimate "2-5" \
    --license "cc-by-sa" \
    --metadata-file "./childes_metadata.json" \
    --preprocess \
    --preprocessor-type transcript

# Example 5: Use LLM filtering for quality control
python main_pipeline.py \
    --language spa \
    --data-source "SpanishWebTexts" \
    --category "child-available-speech" \
    --texts-dir "./spanish_web_texts" \
    --script latin \
    --age-estimate "8-14" \
    --license "cc-by" \
    --preprocess \
    --preprocessor-type llm \
    --llm-model "llama3.2" \
    --llm-prompt "Is this text appropriate and educational for Spanish-learning children aged 8-14? Consider vocabulary level, content appropriateness, and educational value. Respond with JSON: {\"score\": 0-1, \"reason\": \"explanation\"}" \
    --llm-filter-threshold 0.8

# Example 6: Process OpenSubtitles using the dedicated script
python process_opensubtitles.py \
    --language af \
    --script latin \
    --batch-size 100 \
    --upload \
    --repo-id "bhargavns/test"

# Example 7: Chain preprocessing - first extract, then additional processing
# Step 1: Extract texts
python process_opensubtitles.py \
    --language af \
    --script latin \
    --no-preprocess \
    --keep-zip

# Step 2: Apply custom preprocessing
python main_pipeline.py \
    --language af \
    --data-source "OpenSubtitles" \
    --category "subtitles" \
    --texts-dir "./output/preprocessed_texts" \
    --script latin \
    --age-estimate "n/a" \
    --license "cc-by-3.0" \
    --preprocess \
    --preprocessor-type subtitle \
    --remove-timestamps \
    --upload \
    --repo-id "bhargavns/test"