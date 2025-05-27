#!/bin/bash
# example_usage.sh - Examples of how to use the pipeline

# Example 1: Process OpenSubtitles for Afrikaans
python main_pipeline.py opensubtitles \
    --language afr \
    --script latin \
    --age-estimate "n/a" \
    --license "cc-by" \
    --batch-size 100 \
    --source-url "https://opus.nlpl.eu/OpenSubtitles.php" \
    --misc '{"version": "v2024", "preprocessing": "lowercased"}' \
    --upload \
    --repo-id "yourusername/babylm-afr"

# Example 2: Process custom CHILDES data for Dutch
python main_pipeline.py custom \
    --language nld \
    --script latin \
    --age-estimate "2-5" \
    --license "cc-by-sa" \
    --data-source "CHILDES-Dutch" \
    --category "child-directed-speech" \
    --texts-dir "./childes_dutch_texts" \
    --metadata-file "./childes_dutch_metadata.json" \
    --source-url "https://childes.talkbank.org/access/Dutch/" \
    --upload \
    --repo-id "yourusername/babylm-nld"

# Example 3: Process educational content for German without uploading
python main_pipeline.py custom \
    --language deu \
    --script latin \
    --age-estimate "6-12" \
    --license "cc-by" \
    --data-source "German-Educational-Texts" \
    --category "educational" \
    --texts-dir "./german_edu_texts" \
    --misc '{"grade_level": "primary", "subject": "mixed"}'

# Example 4: Process Wikipedia for kids in Spanish
python main_pipeline.py custom \
    --language spa \
    --script latin \
    --age-estimate "8-14" \
    --license "cc-by-sa" \
    --data-source "Vikidia-Spanish" \
    --category "child-wiki" \
    --texts-dir "./vikidia_spanish_texts" \
    --source-url "https://es.vikidia.org/" \
    --source-identifier "vikidia-es-2024-01"


python main_pipeline.py opensubtitles \
    --language af \
    --script latin \
    --age-estimate "n/a" \
    --license "cc-by" \
    --batch-size 100 \
    --source-url "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/xml/{lang}.zip" \
    --misc '{"version": "v2024", "preprocessing": "lowercased"}' \