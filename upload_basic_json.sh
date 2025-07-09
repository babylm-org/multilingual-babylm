#!/bin/bash

# Basic command to upload a babyLM dataset from a JSON file
# Recommended options included for: 
# - language filtering
# - text preprocessing
# For more data loading options, see the example_usage.sh file
python pipeline.py \
    --language eng\
    --script Latn \
    --data-path "examples/dataset.json" \
    --data-type "json" \
    --preprocess-text \
    --enable-language-filter \
    --upload \
    --repo-id "username/babylm-eng"
    #--pad-opensubtitles 
    # if you want to pad your dataset with OpenSubtitles data
